import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DensityEstimate(nn.Module):
    def __init__(self):
        super(DensityEstimate, self).__init__()
        pass

    def fit(self, samples):
        pass

    def sample(self):
        pass


class NormalizingFlow(DensityEstimate):
    def __init__(self, seed, num_flows, prior, d=2, h=100, lr=1e-3, dequantization=True, decay_every=5000, lr_decay_rate=0.5, weight_decay=0.0):
        super(NormalizingFlow, self).__init__()

        print('RealNVP by JT.')
        torch.manual_seed(seed)
        self.dequantization = dequantization
        self.decay_every = decay_every

        # scale (s) network
        nets = lambda: nn.Sequential(nn.Linear(d // 2, h), nn.LeakyReLU(),
                                     nn.Linear(h, h), nn.LeakyReLU(),
                                     nn.Linear(h, d // 2), nn.Tanh())

        # translation (t) network
        nett = lambda: nn.Sequential(nn.Linear(d // 2, h), nn.LeakyReLU(),
                                     nn.Linear(h, h), nn.LeakyReLU(),
                                     nn.Linear(h, d // 2))

        self.prior = prior # MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device))

        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows
        self.d = d
        self.params = [
            {'params': self.t.parameters(), 'weight_decay': weight_decay},
            {'params': self.s.parameters(), 'weight_decay': weight_decay},
        ]

        self.optimizer = torch.optim.Adamax(self.params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_decay_rate)

    def coupling(self, x, index, forward=True):
        # x: input, either images (for the first transformation) or outputs from the previous transformation
        # index: it determines the index of the transformation
        # forward: whether it is a pass from x to y (forward=True), or from y to x (forward=False)

        (xa, xb) = torch.chunk(x, 2, 1)

        s = self.s[index](xa)
        t = self.t[index](xa)

        if forward:
            # yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            # xb = f(y)
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)
        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)
        return x

    def forward(self, x, reduction='avg'):
        z, log_det_J = self.f(x)
        if reduction == 'sum':
            return -(self.prior.log_prob(z) + log_det_J).sum()
        else:
            return -(self.prior.log_prob(z) + log_det_J).mean()

    def sample(self, batch_size):
        z = self.prior.sample((batch_size, self.d)).float()
        z = z[:, 0, :]
        x = self.f_inv(z)
        return x.view(-1, self.d)

    def log_prob(self, x):
        return -self.forward(x, reduction='sum')  # log p(x) = log p(z) + log det J

def flow_evaluation(model, test_loader, epoch=None):
    # EVALUATION

    model.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        if model.dequantization:
            test_batch = test_batch + (1. - torch.rand(test_batch.shape)).to(device) / 2.
        loss_t = model.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N += test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}, lr={model.scheduler.get_lr()}')

    return loss

def flow_fit(model, task_id, num_epochs, training_loader, val_loader, writer, name, early_stopping=False):
    nll_val = []
    best_nll = 3000.
    patience = 0
    max_patience = 40 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
    # Main loop
    iter = 0
    for e in range(num_epochs):

        model.train()
        for indx_batch, batch in enumerate(training_loader):
            batch = batch.to(device)
            if model.dequantization:
                batch = batch + (1. - torch.rand(batch.shape).to(device)) / 2.
            loss = model.forward(batch)

            model.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            model.optimizer.step()

            if (iter + 1) % model.decay_every == 0:
               model.scheduler.step()

            iter += 1

        # Validation
        loss_val = flow_evaluation(model, val_loader, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        writer.add_scalar("task_" + str(task_id) + "_train_nll", loss.detach().item(), e)
        writer.add_scalar("task_" + str(task_id) + "_val_nll", loss_val, e)
        writer.add_scalar("task_" + str(task_id) + "_lr", model.scheduler.get_lr()[0], e)

        if early_stopping:
            model_pth = os.path.join('models', name + '.pth')
            if e == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict()},
                    model_pth,
                )
                best_nll = loss_val
            else:
                if loss_val < best_nll:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict()},
                        model_pth,
                    )
                    best_nll = loss_val
                    patience = 0
                else:
                    patience += 1

            if patience > max_patience and e > 200:
                checkpoint = torch.load(model_pth)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                loss_val = flow_evaluation(model, val_loader, epoch=e + 1)
                writer.add_scalar("task_" + str(task_id) + "_val_nll", loss_val, e + 1)  # final val loss after loading best model
                break

    nll_val = np.asarray(nll_val)

    return nll_val


class MixtureOfGaussians(DensityEstimate):
    def __init__(self, n_components, max_iter=100, n_init=1):
        super().__init__()
        self.n_components = n_components
        self.gm = GaussianMixture(n_components=n_components, max_iter=max_iter, n_init=n_init, random_state=0)

    def fit(self, samples):
        self.gm.fit(samples)

    def sample(self, n_samples):
        return self.gm.sample(n_samples)

    def goodness_of_fit(self, samples):
        return self.gm.score(samples)

    def sklearn_2_pytorch(self, d):
        weights = torch.from_numpy(self.gm.weights_).to(device)
        self.weights = nn.Parameter(weights.clone().detach().requires_grad_(True))
        assert weights.shape[0] == self.n_components
        means = torch.from_numpy(self.gm.means_).to(device)
        self.means = nn.Parameter(means.clone().detach().requires_grad_(True))
        assert means.shape == (self.n_components, d)
        covs = torch.from_numpy(self.gm.covariances_).to(device)
        self.covs = nn.Parameter(covs.clone().detach().requires_grad_(True))
        assert covs.shape == (self.n_components, d, d)
        self.mix = Categorical(self.weights)
        self.comp = MultivariateNormal(self.means, self.covs)
        self.gmm = MixtureSameFamily(self.mix, self.comp)

    def log_prob(self, w):
        # w is numpy array like of shape (n_samples, n_features)
        # thus n_features is the dimensionality of the BNN we are working with
        # returns tensor of size (n_samples, )
        assert len(w.shape) == 2
        return self.gmm.log_prob(w)

    def log_prob_(self, w):
        return self.gm.score_samples(w)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var

class VAE(DensityEstimate):
    def __init__(self, x_dim=784):
        batch_size = 100
        hidden_dim = 400
        self.latent_dim = 200
        lr = 1e-3

        self.encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=self.latent_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

        self.params = [
            {'params': self.encoder.parameters(), 'weight_decay': 0.0},
            {'params': self.decoder.parameters(), 'weight_decay': 0.0},
        ]

        self.optimizer = Adam(self.params, lr=lr)

    def elbo(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var

    def log_prob(self, w):
        pass

    def sample(self, batch_size):
        with torch.no_grad():
            noise = torch.randn(batch_size, self.latent_dim).to(device)
            x_hat = self.decoder(noise)
        return x_hat


def train_vae(model, epochs, train_loader, batch_size, x_dim):
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            model.optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = model.elbo(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            model.optimizer.step()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
    model.eval()
    print("Finish!!")