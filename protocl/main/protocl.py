import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from copy import deepcopy
from protocl.main.encoders import get_encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ProtoCL(nn.Module):
    """
    Implementation based off of PCOC from:
    @article{harrison2019continuous,
    title={Continuous meta-learning without tasks},
    author={Harrison, James and Sharma, Apoorva and Finn, Chelsea and Pavone, Marco},
    journal={arXiv preprint arXiv:1912.08866},
    year={2019}
    }
    """
    def __init__(self, config):
        super().__init__()

        self.config = deepcopy(config)
        self.x_dim = config['model.x_dim']
        self.phi_dim = config['model.phi_dim']
        self.y_dim = config['model.y_dim']

        self.sigma_eps = np.zeros(
            [self.y_dim, 1]) + np.asarray(eval(config['model.sigma_eps'])
        )
        self.cov_dim = self.sigma_eps.shape[-1] # 1
        print("Using %d parameters in covariance:" % self.cov_dim)
        if self.phi_dim % self.cov_dim != 0:
            raise ValueError("cov_dim must evenly divide phi_dim")

        self.logSigEps = nn.Parameter(
            torch.from_numpy(np.log(self.sigma_eps)),
            requires_grad=self.config['train.learnable_noise'],
        )

        Linv_offset = config['model.Linv_init']
        dir_scale = config['model.dirichlet_scale']
        
        self.Q = nn.Parameter(
            torch.randn(self.y_dim, self.cov_dim, self.phi_dim // self.cov_dim)
        )
        self.logLinv = nn.Parameter(
            torch.randn(self.y_dim, self.cov_dim) + Linv_offset
        )
        
        self.log_dirichlet_priors = nn.Parameter(
            dir_scale * torch.ones(self.y_dim),
            requires_grad=config['train.learnable_dirichlet'],
        )

        self.normal_nll_const = self.phi_dim * np.log(2 * np.pi)

        self.encoder = get_encoder(config).to(device)

        params = [
            {'params': self.encoder.parameters()},
        ]

        if self.config['train.learnable_noise']:
            params.append({'params': [self.logSigEps]})

        if self.config['train.learnable_dirichlet']:
            params.append({'params': [self.log_dirichlet_priors]})

        self.optimizer = optim.Adam(
            params,
            lr=config['train.learning_rate'],
            weight_decay=config['train.weight_decay'],
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1, 
            gamma=0.5,
        )

    @property
    def invSigEps(self):
        return torch.exp(-self.logSigEps)  # .repeat(self.y_dim,1)

    @property
    def SigEps(self):
        return torch.exp(self.logSigEps)  # .repeat(self.y_dim,1)

    def prior_params(self):
        Q0 = self.Q
        Linv0 = torch.exp(self.logLinv)
        dir_weights = torch.exp(self.log_dirichlet_priors)
        return Q0, Linv0, dir_weights

    def set_params(self, params):
        self.Q = nn.Parameter(params[0])
        self.logLinv = nn.Parameter(torch.log(params[1]))
        self.log_dirichlet_priors = nn.Parameter(
            torch.log(params[2]),
            requires_grad=self.config['train.learnable_dirichlet'],
        )

    def recursive_update(self, phi, y, params):
        """
            inputs: phi: shape (..., cov_dim, k )
                    y:   shape (..., y_dim )
                    params: tuple of Q, Linv, dir_weights
                        Q: shape (..., y_dim, cov_dim, k)
                        Linv: shape (..., y_dim, cov_dim)
                        dir_weights: shape (..., y_dim)
        """
        Q, Linv, dir_weights = params

        # zeros out entries all except class y
        invSigEps_masked = self.invSigEps * y.squeeze(1).unsqueeze(-1)

        Q = Q + invSigEps_masked.unsqueeze(-1) * phi
        Linv = Linv + invSigEps_masked
        dir_weights = dir_weights + y
        return (Q, Linv, dir_weights)

    def log_predictive_prob(self, x, y, posterior_params):
        """
            input:  x: shape (..., x_dim)
                    y: shape (..., y_dim)
                    posterior_params: tuple of Q, Linv:
                        Q: shape (..., y_dim, cov_dim, k)
                        Linv: shape (..., y_dim, cov_dim)
                        dir_weights: shape (..., y_dim)
            output: logp: log p(y, x | posterior_params) (..., y_dim)
                    updated_params: updated posterior params after factoring in (x,y) pair
        """

        x_shape = list(x.shape)

        if len(x_shape) > 4:  # more than one batch dim
            x = x.reshape([-1] + x_shape[-3:])

        phi = self.encoder(x)  # (..., phi_dim)
        if len(x_shape) > 4:
            phi = phi.reshape(x_shape[:-3] + [self.phi_dim])

        Q, Linv, dir_weights = posterior_params 
        mu = Q / Linv.unsqueeze(-1) # (1, 1, y_dim, 1, phi_dim)
        pred_cov = 1. / Linv + self.SigEps  # (..., cov_dim, y_dim)

        phi_shape = phi.shape # (b, phi_dim)
        phi_reshaped = phi.reshape(*(list(phi_shape)[:-1] + [1, self.cov_dim, -1]))  # (..., 1, cov_dim, k)

        err = phi_reshaped - mu # (b, y_dim, 1, phi_dim)

        nll_quadform = (err ** 2 / pred_cov.unsqueeze(-1)).sum(-1).sum(-1)
        nll_logdet = (self.phi_dim / self.cov_dim) * torch.log(pred_cov).sum(-1) # (b, 1, y_dim) sum of log of diagonal entries

        logp = -0.5 * (nll_quadform + nll_logdet + self.normal_nll_const)  # log p(x | y)

        logp += torch.log(dir_weights / dir_weights.sum(-1, keepdim=True))  # multiply by p(y) posterior to get p(x, y)

        posterior_params = [p.detach() for p in posterior_params]
        updated_params = self.recursive_update(phi_reshaped, y, posterior_params)
        updated_params = [p.detach() for p in updated_params]
        self.set_params(updated_params)

        return logp, updated_params


    def nll(self, log_pi):
        """
            log_pi: shape(batch_size x t x ...)
            log_prgx: shape (batch_size x t x ...)
        """
        return -torch.logsumexp(log_pi, dim=1)

    def log_posterior(self, x_mat, y_mat):
        """
        Takes in x,y batches; recursively compute posteriors
        Inputs:
        - x_mat; shape = batch size x x_dim
        - y_mat; shape = batch size x y_dim
        """

        # define initial params and append to list
        # we add a batch dimension if its not already there
        # prior_params = tuple(p[None, ...] if len(p.shape) < 4 for p in self.prior_params()) # (Q0, Linv0, dir_weights)
        prior_params = list(self.prior_params())
        prior_params[0] = prior_params[0][None, ...] if len(prior_params[0].shape) == 3 else prior_params[0]
        prior_params[1] = prior_params[1][None, ...] if len(prior_params[1].shape) == 2 else prior_params[1]
        prior_params[2] = prior_params[2][None, ...] if len(prior_params[2].shape) == 1 else prior_params[2]

        # if classification, log_pi == p(y,x|eta) for all y (batchsize, y_dim)
        log_p, updated_posterior_params = self.log_predictive_prob(
            x_mat,
            y_mat,
            prior_params,
        )

        # normalize to get p(y | x) # (batchsize, y_dim)
        nll = -nn.functional.log_softmax(log_p.squeeze(1), dim=-1) # (..., y_dim)

        return updated_posterior_params, nll

    def forward(self, x, posterior_params):
        """
            input: x, posterior params
            output: log p(x | y) for all y
        """
        x_shape = list(x.shape)

        if len(x_shape) > 4:  # more than one batch dim
            x = x.reshape([-1] + x_shape[-3:])

        phi = self.encoder(x)  # (..., phi_dim)
        if len(x_shape) > 4:
            phi = phi.reshape(x_shape[:-3] + [self.phi_dim])

        Q, Linv, dir_weights = posterior_params
        mu = Q / Linv.unsqueeze(-1)  # (..., y_dim, cov_dim, k)
        pred_cov = 1. / Linv + self.SigEps()  # (..., y_dim, cov_dim)

        phi_shape = phi.shape
        phi_reshaped = phi.reshape(*(list(phi_shape)[:-1] + [self.cov_dim, -1]))  # (..., cov_dim, k)

        err = phi_reshaped.unsqueeze(-3) - mu  # (..., y_dim, cov_dim, k)

        nll_quadform = (err ** 2 / pred_cov.unsqueeze(-1)).sum(-1).sum(-1)
        nll_logdet = (self.phi_dim / self.cov_dim) * torch.log(pred_cov).sum(-1)  # sum of log of diagonal entries

        logp = -0.5 * (nll_quadform + nll_logdet + self.normal_nll_const)  # log p(x | y)

        logp += torch.log(dir_weights / dir_weights.sum(-1, keepdim=True))  # multiply by p(y) to get p(x, y)

        return logp
