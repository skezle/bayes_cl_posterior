import datetime
import time
import os
import numpy as np
import copy
import torch
import pickle
from argparse import ArgumentParser


import hamiltorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.extend(['../'])
from protocl.main.utils import scores_to_arr
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal

import pyro
import pyro.ops.stats as stats

from data import SplitMnistGenerator, PermutedMnistGenerator, ToyGaussiansGenerator, ToyGaussiansContGenerator
from nets import weights_init_normal, Net
from vis import plot_2d
from density_estimators import MixtureOfGaussians, NormalizingFlow, flow_fit

from pytorch_model_summary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sampler:
    def __init__(
        self, 
        seed, 
        data_gen,
        toy_cont_gaussians,
        hid_dim,
        batch_size,
        n_chains,
        n_samples,
        tau,
        mixture_of_gaussians, 
        max_n_comps, 
        flow,
        num_flows,
        flow_dim,
        flow_decay_every,
        flow_early_stopping,
        flow_weight_decay, 
        flow_lr, 
        tag,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        hamiltorch.set_random_seed(seed)

        self.batch_size = batch_size
        self.data_gen = data_gen
        self.toy_cont_gaussians = toy_cont_gaussians
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.max_n_comps = max_n_comps
        self.flow = flow
        self.mixture_of_gaussians = mixture_of_gaussians
        self.num_flows = num_flows
        self.flow_dim = flow_dim  # the number of neurons in scale (s) and translation (t) nets
        self.flow_decay_every = flow_decay_every
        self.flow_early_stopping = flow_early_stopping
        self.flow_weight_decay = flow_weight_decay
        self.flow_lr = flow_lr
        self.tau = tau
        self.tag = tag

        in_dim, out_dim = self.data_gen.get_dims()
        self.n_tasks = 1 if one_task else self.data_gen.max_iter

        self.model = Net(seed, in_dim, out_dim, hid_dim)

        # TB summary
        DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        tensorboard_dir = os.path.join(DIR_PATH, 'logs', str(timestamp) + tag)
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)

    def train(
        self, 
        train_loader,
        params_init, 
        tau_list, 
        prior, 
        previous_hmc_params,
    ):
        self.model.train()
        params_hmc_lst = []

        model_copy = type(self.model)(self.model.seed, self.model.in_dim, self.model.out_dim, self.model.hid_dim)  # get a new instance
        model_copy.load_state_dict(self.model.state_dict())  # copy weights and stuff

        for i in range(self.n_chains):

            print("\n\t Chain {} \n".format(i))

            # for each new chain load the copy
            # change the random seed for each new init
            model = type(model_copy)(self.model.seed + i, self.model.in_dim, self.model.out_dim, self.model.hid_dim)
            model.load_state_dict(model_copy.state_dict())
            model.apply(weights_init_normal)

            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 1:
                    break
                print("Iter {}".format(batch_idx + 1))

                x_train, y_train = data.to(device), target.to(device)  # (..., dim), (...)

                step_size = 0.001  # 0.01# 0.003#0.002
                L = 20  # 3
                tau_out = 1.
                normalizing_const = 1.
                burn = 1000 #GPU: 3000

                params_hmc, acc_rate = hamiltorch.sample_model(
                    model,
                    x_train,
                    y_train,
                    params_init=params_init,
                    model_loss='multi_class_linear_output',
                    num_samples=self.n_samples,
                    burn=burn,
                    step_size=step_size,
                    num_steps_per_sample=L,
                    tau_out=tau_out,
                    tau_list=tau_list,
                    normalizing_const=normalizing_const,
                    prior=prior,
                    debug=2,
                    previous_params=previous_hmc_params,
                )
                # sample every 5 from the chain
                params_hmc_lst.append(params_hmc[::5])

        params_hmc_tensor = torch.stack([torch.stack(p) for p in params_hmc_lst])
        del params_hmc_lst

        out = stats.effective_sample_size(params_hmc_tensor.cpu(), chain_dim=0, sample_dim=1)

        return params_hmc_tensor, np.mean(out.numpy()), acc_rate

    def test(
        self, 
        test_loader,
        params_hmc,
        tau_list,
        tag='',
        prior=None,
        writer=None
    ):
        acc = torch.zeros(int(len(params_hmc)) - 1)
        nll = torch.zeros(int(len(params_hmc)) - 1)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx >= 1:
                    break

                x_test, y_test = data.to(device), target.to(device)
                pred_list, _ = hamiltorch.predict_model(
                    model=self.model,
                    x=x_test,
                    y=y_test,
                    samples=params_hmc,
                    model_loss='multi_class_log_softmax_output',
                    tau_out=1.,
                    tau_list=tau_list,
                    prior=prior,
                )
                _, pred = torch.max(pred_list, 2)
                ensemble_proba = F.softmax(pred_list[0], dim=-1)
                for s in range(1, len(params_hmc)):
                    _, pred = torch.max(pred_list[:s].mean(0), -1)
                    acc[s - 1] = (pred.float() == y_test.flatten()).sum().float() / y_test.shape[0]
                    ensemble_proba += F.softmax(pred_list[s], dim=-1)
                    nll[s - 1] = F.nll_loss(
                        torch.log(ensemble_proba.cpu() / (s + 1)),
                        y_test[:].long().cpu().flatten(),
                        reduction='mean'
                    )
                    if writer is not None:
                        writer.add_scalar("task_{}_chain_convergence/acc".format(tag), acc[s - 1], s)
                        writer.add_scalar("task_{}_chain_convergence/nll".format(tag), nll[s - 1], s)

        return acc, nll

    def run(self):
        """Continual Learning training loop"""

        test_loaders = []
        ess_lst = []

        all_acc, all_nll = np.array([]), np.array([])

        params_init = hamiltorch.util.flatten(self.model).to(device).clone()
        print(params_init.shape)
        tau_list = []
        # tau = tau #10.  # ./100. # 1/50
        for w in self.model.parameters():
            #     print(w.nelement())
            #     tau_list.append(tau/w.nelement())
            tau_list.append(self.tau)
        tau_list = torch.tensor(tau_list).to(device)

        d = params_init.shape[0]
        prior = None
        previous_hmc_params = None # TODO: remove this when flow debug has finished

        for task_id in range(self.n_tasks):
            print("\n\t Task {} \n".format(task_id + 1))

            plot_tag = self.tag + '_task_' + str(task_id + 1)
            # Plot visualisation in 2D for the toy dataset
            x_train, y_train, x_test, y_test, x_test_full, y_test_full = self.data_gen.next_task()
            test_full_loader = DataLoader(
                TensorDataset(torch.Tensor(x_test_full), 
                torch.Tensor(y_test_full).long()),
                len(x_test_full),
                shuffle=False,
            )

            train_loader = DataLoader(
                TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long()),
                self.batch_size,
                shuffle=True,
            )

            test_loader = DataLoader(
                TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long()), 
                self.batch_size,
                shuffle=False
            )

            test_loaders.append(copy.deepcopy(test_loader))

            # print("Using cached HMC params")
            # with open('results/hmc_samples_{0}_task{1}.pickle'.format(load_hmc_params_tag, task_id + 1), 'rb') as file:
            #     d = pickle.load(file)
            #     hmc_params = d['hmc_params']
            # ess, acc_rate = np.nan, np.nan
            # hmc_params = torch.from_numpy(hmc_params).float().to(device)
            hmc_params, ess, acc_rate = self.train(
                train_loader,
                params_init,
                tau_list,
                prior=prior,
                previous_hmc_params=previous_hmc_params,
            )  # hmc params is of size (n_chains, n_samples)
            ess_lst.append(ess)

            # if cache_hmc_samples:
            #     with open('results/hmc_samples_{0}_task{1}.pickle'.format(tag, task_id + 1), 'wb') as file:
            #         pickle.dump({'hmc_params': hmc_params.cpu().numpy()}, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Access convergence
            _, _ = self.test(
                test_loader,
                hmc_params[0],  # only pass in one chain for eval
                tau_list,
                prior=prior,
                tag=str(task_id + 1),
                writer=self.writer,
            )

            # Test accuracy on all previous tasks
            accs, nlls = [], []
            for tl in test_loaders:
                acc, nll = self.test(
                    tl,
                    hmc_params[-1],  # only pass in one chain for eval
                    tau_list,
                    prior=prior,
                    tag=plot_tag,
                    writer=None,
                )
                accs.append(acc[-1])
                nlls.append(nll[-1])
            
            all_acc = scores_to_arr(all_acc, accs)
            all_nll = scores_to_arr(all_nll, nlls)

            self.writer.add_scalar("ess", ess, task_id + 1)
            self.writer.add_scalar("acc_rate", acc_rate, task_id + 1)
            self.writer.add_scalar("acc", acc[-1], task_id + 1)
            self.writer.add_scalar("nll", nll[-1], task_id + 1)

            if self.toy_cont_gaussians:
                plot_2d(self.model, hmc_params[-1], tau_list, prior, test_full_loader, data_gen, task_id, self.tag)

            # Propagate Posterior

            # Split HMC samples into a train and validation
            hmc_params_arr = hmc_params.cpu().numpy().reshape(-1, d)  # (n_chains, ..., d) --> (n_chains x ..., d)
            N = hmc_params_arr.shape[0]
            perm_inds = list(range(N))
            np.random.shuffle(perm_inds)  # shuffles in-place do not re-assign

            if self.mixture_of_gaussians:
                # Iterate over different number of components to the GMM and see which maximises the ll
                lls = []
                n_comps = [int(n) for n in np.geomspace(1, self.max_n_comps, 20)]
                for n in n_comps:
                    posterior = MixtureOfGaussians(n_components=n, max_iter=500, n_init=5)
                    posterior.fit(hmc_params_arr[perm_inds[:int(0.75 * N)], :])
                    ll = posterior.goodness_of_fit(hmc_params_arr[perm_inds[int(0.75 * N):], :])
                    lls.append(ll)
                    self.writer.add_scalar("task" + str(task_id) + "mog_ll", ll, n)

                # Fit GMM to optimal number of components
                opt_n_comp = n_comps[np.argmax(lls)]
                print("Optimal number of components: {}".format(opt_n_comp))
                posterior = MixtureOfGaussians(n_components=opt_n_comp)
                posterior.fit(hmc_params_arr[perm_inds[:int(0.75 * self.n_samples)], :])
                posterior.sklearn_2_pytorch(d)
                self.writer.add_scalar("mog_opt_n_comps", opt_n_comp, task_id + 1)

                # generate samples from GMM and assess the accuracy of the BNN
                samples = posterior.sample(n_samples=32)
                samples = torch.from_numpy(samples[0]).float().to(device)
                acc, nll = self.test(
                    test_loader,
                    samples,
                    tau_list,
                    prior=prior,
                    tag=plot_tag,
                )
                self.writer.add_scalar("gmm_sampled_acc", np.mean(acc.numpy()), task_id + 1)
                self.writer.add_scalar("gmm_sampled_nll", np.mean(nll.numpy()), task_id + 1)

            if self.flow:
                # define flow parameters
                d = params_init.shape[0]  # input dim, the number of parameters
                if self.mixture_of_gaussians:
                    print("Using GMM as initialization to flow density estimator.")
                    prior = posterior.gmm # Using the GMM as a prior for the flow
                else:
                    prior = MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device))

                num_epochs = 200  # max. number of epochs for flow training

                # initialize RealNVP flow
                posterior = NormalizingFlow(
                    seed=seed,
                    num_flows=self.num_flows,
                    prior=prior,
                    d=d,
                    h=self.flow_dim,
                    lr=self.flow_lr,
                    dequantization=False,
                    decay_every=self.flow_decay_every,
                    weight_decay=self.flow_weight_decay,
                ).to(device)

                # summary of the model
                print(summary(posterior, torch.zeros(1, d).to(device), show_input=True, show_hierarchical=False))

                training_loader = DataLoader(
                    hmc_params_arr[perm_inds[:int(0.75 * N)], :],
                    batch_size=32,
                    shuffle=True,
                )

                val_loader = DataLoader(
                    hmc_params_arr[perm_inds[int(0.75 * N):], :],
                    batch_size=32,
                    shuffle=True,
                )

                # fit flow parameters to the posterior samples from our HMC chain
                _ = flow_fit(
                    model=posterior,
                    task_id=task_id,
                    num_epochs=num_epochs,
                    training_loader=training_loader,
                    val_loader=val_loader,
                    writer=self.writer,
                    name="{0}_task{1}".format(self.tag,     task_id + 1),
                    early_stopping=self.flow_early_stopping,
                )

                # generate samples from flow and assess the accuracy of the BNN
                samples = posterior.sample(batch_size=24)
                acc, nll = self.test(
                    test_loader,
                    samples,
                    tau_list,
                    prior=prior,
                    tag=plot_tag,
                )
                self.writer.add_scalar("flow_sampled_acc", np.mean(acc.numpy()), task_id + 1)
                self.writer.add_scalar("flow_sampled_nll", np.mean(nll.numpy()), task_id + 1)

            # use posterior as prior for new task
            if self.flow or self.mixture_of_gaussians:
                prior = posterior
            else:
                prior = None

        self.writer.close()

        return all_acc, all_nll, ess_lst


if __name__ == "__main__":
    parser = ArgumentParser()

    # CL setup
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--multiclass', dest='multiclass', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='split_mnist',
                        help="data set name (default: {:s})".format('split_mnist'))
    parser.add_argument('--dataset_sz', type=int, default=800)
    parser.add_argument('--permuted', dest='permuted', action='store_true', default=False)
    parser.add_argument('--fashion', dest='fashion', action='store_true', default=False)
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')
    parser.add_argument('--one_task', dest='one_task', action='store_true', default=False)

    # Posterior propagation params
    parser.add_argument('--mixture_of_gaussians', dest='mixture_of_gaussians', action='store_true', default=False)
    parser.add_argument('--max_n_comps', type=int, default=100, help='maximum number of MoG components to search over.')
    parser.add_argument('--flow', dest='flow', action='store_true', default=False)
    parser.add_argument('--cache_hmc_samples', dest='cache_hmc_samples', action='store_true', default=False,
                        help='whether to cache the hmc samples to then tune the flow.')
    parser.add_argument('--num_flows', type=int, default=4, help='number of flows in the RealNVP.')
    parser.add_argument('--flow_dim', type=int, default=100, help='hidden state dim of the flow.')
    parser.add_argument('--flow_lr', type=float, default=1e-1, help='flow learning rate.')
    parser.add_argument('--flow_weight_decay', type=float, default=0.0, help='flow weight decay.')
    parser.add_argument('--flow_decay_every', type=int, default=5000, help='learning rate decay for flow.')
    parser.add_argument('--flow_early_stopping', dest='flow_early_stopping', action='store_true', default=False, help='Whether to train the flow with early stopping.')

    # BNN params
    parser.add_argument('--hid_dim', type=int, default=80)
    parser.add_argument('--tau', type=float, default=10.0, help='Gaussian prior precision.')

    # HMC params
    parser.add_argument('--n_chains', type=int, default=1)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=2500)

    args = parser.parse_args()
    params = vars(args)

    print(params)

    assert not (params['permuted'] and params['multiclass'])

    n_tasks = 5
    one_task = params['one_task']
    hid_dim = params['hid_dim']
    tau = params['tau']
    runs = params['n_runs']
    n_chains = params['n_chains']

    accs, nlls = np.zeros((runs, n_tasks, n_tasks)), np.zeros((runs, n_tasks, n_tasks))
    ess_all = np.zeros((runs, n_tasks))

    for i in range(runs):
        seed = params['seed'] + i
        if params['dataset'] == 'toy_gaussians':
            dataset_size = params['dataset_sz']
            data_gen = ToyGaussiansGenerator(N=dataset_size, cl3=False, flatten_labels=True)
            toy_cont_gaussians = False
        elif params['dataset'] == 'toy_continuous_gaussians':
            dataset_size = params['dataset_sz']
            data_gen = ToyGaussiansContGenerator(num_samples=dataset_size, flatten_labels=True)
            toy_cont_gaussians = True
        else:
            raise ValueError


        sampler = Sampler(
            seed=seed,
            data_gen=data_gen, 
            toy_cont_gaussians=toy_cont_gaussians,
            hid_dim=hid_dim,
            batch_size=dataset_size,
            n_chains=n_chains,
            n_samples=params['n_samples'],
            tau=tau,
            mixture_of_gaussians=params['mixture_of_gaussians'],
            max_n_comps=params['max_n_comps'],
            flow=params['flow'],
            num_flows=params['num_flows'], 
            flow_dim=params['flow_dim'],
            flow_decay_every=params['flow_decay_every'], 
            flow_early_stopping=params['flow_early_stopping'],
            flow_weight_decay=params['flow_weight_decay'],
            flow_lr=params['flow_lr'],
            tag=params['tag'] + "_run_" + str(i),
        )

        acc, nll, ess = sampler.run()

        accs[i, :, :] = acc
        nlls[i, :, :] = nll
        ess_all[i, :] = ess

    with open('results/hmc_res_{}.pickle'.format(params['tag']), 'wb') as handle:
        pickle.dump({'accs': accs, 'nlls': nlls, 'ess': ess_all}, handle, protocol=pickle.HIGHEST_PROTOCOL)
