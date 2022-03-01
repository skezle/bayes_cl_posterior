import datetime
import time
import os
import numpy as np
import copy
import torch
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import hamiltorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from protocl.main.utils import scores_to_arr
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal

import pyro
import pyro.ops.stats as stats

from data import SplitMnistGenerator, PermutedMnistGenerator, ToyGaussiansGenerator, ToyGaussiansContGenerator
from density_estimators import MixtureOfGaussians, NormalizingFlow, flow_evaluation, flow_fit

from pytorch_model_summary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## thanks: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, seed, in_dim, out_dim, hid_dim, cl3=False):
        super(Net, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.task_idx = 1
        self.cl3 = cl3

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        return output

def plot_2d(model, params_hmc, tau_list, prior, test_full_loader, datagen, task_idx, tag):
    # Plot visualisation (2D figure)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_full_loader):
            if batch_idx >= 1:
                break

            x_test, y_test = data.to(device), target.to(device)
            pred_list, log_prob_list = hamiltorch.predict_model(
                model=model,
                x=x_test,
                y=y_test,
                samples=params_hmc,
                model_loss='multi_class_log_softmax_output',
                tau_out=1.,
                tau_list=tau_list,
                prior=prior,
            )
            ensemble_proba = F.softmax(pred_list.mean(0), dim=-1)
            _, pred = torch.max(pred_list.mean(0), -1)
            acc = (pred.float() == y_test.flatten()).sum().float() / y_test.shape[0]
            print("acc: {}".format(acc))

    print(ensemble_proba.shape)
    pred_y_show = ensemble_proba[:, 0] - ensemble_proba[:, 1]
    # cl_outputs, _ = torch.max(cl_outputs, dim=-1)
    # cl_show = 2 * cl_outputs - 1

    pred_y_show = pred_y_show.detach()
    # if use_cuda:
    #     cl_show = cl_show.cpu()
    pred_y_show = pred_y_show.cpu().numpy()
    cl_show = pred_y_show.reshape(datagen.test_shape)
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plt.figure()
    axs = plt.subplot(111)
    axs.title.set_text('HMC SH')
    im = plt.imshow(cl_show, cmap='Blues',
                    extent=(datagen.x_min, datagen.x_max, datagen.y_min, datagen.y_max), origin='lower')
    for l in range(task_idx + 1):
        idx = np.where(datagen.y == l)
        plt.scatter(datagen.X[idx][:, 0], datagen.X[idx][:, 1], c=color[l], s=0.1)
        idx = np.where(datagen.y == l + datagen.offset)
        plt.scatter(datagen.X[idx][:, 0], datagen.X[idx][:, 1], c=color[l + datagen.offset], s=0.1)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.savefig("plots/hmc_toy_gaussians_task{0}_{1}.png".format(task_idx, tag))

def train(model, train_loader, params_init, tau_list, task_id, n_chains=1, n_samples=2500, prior=None, debug=False,
          previous_hmc_params=None):
    model.train()
    params_hmc_lst = []

    model_copy = type(model)(model.seed, model.in_dim, model.out_dim, model.hid_dim)  # get a new instance
    model_copy.load_state_dict(model.state_dict())  # copy weights and stuff

    for i in range(n_chains):

        print("\n\t Chain {} \n".format(i))

        # for each new chain load the copy
        # change the random seed for each new init
        model = type(model_copy)(model.seed + i, model.in_dim, model.out_dim, model.hid_dim)
        model.load_state_dict(model_copy.state_dict())
        model.apply(weights_init_normal)

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 1:
                break
            print("Iter {}".format(batch_idx + 1))

            x_train, y_train = data.to(device), target.to(device)  # (..., dim), (...)

            step_size = 0.001  # 0.01# 0.003#0.002
            # n_samples = n_samples #2000 # 3000
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
                num_samples=n_samples,
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

            params_hmc_lst.append(params_hmc[::5])

    params_hmc_tensor = torch.stack([torch.stack(p) for p in params_hmc_lst])

    del params_hmc_lst
    out = stats.effective_sample_size(params_hmc_tensor.cpu(), chain_dim=0, sample_dim=1)
    print(params_hmc_tensor.shape)

    return params_hmc_tensor, np.mean(out.numpy()), acc_rate


def test(model, test_loader, params_hmc, tau_list, tag='', prior=None, writer=None):
    acc = torch.zeros(int(len(params_hmc)) - 1)
    nll = torch.zeros(int(len(params_hmc)) - 1)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 1:
                break

            x_test, y_test = data.to(device), target.to(device)
            pred_list, log_prob_list = hamiltorch.predict_model(
                model=model,
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


def run(data_gen, seed, one_task, hid_dim, batch_size=500, tau=10., n_chains=1, n_samples=2500,
        mixture_of_gaussians=False, max_n_comps=None, flow=False, tag='', cache_hmc_samples=False,
        n_flows=4, flow_dim=256, flow_decay_every=5000, flow_early_stopping=False, flow_weight_decay=0.0,
        flow_lr=1e-1, debug=False, load_hmc_params_tag=None, toy_gaussians=False):
    """Continual Learning training loop"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    hamiltorch.set_random_seed(seed)

    in_dim, out_dim = data_gen.get_dims()
    test_loaders = []
    ess_lst = []

    all_acc, all_nll = np.array([]), np.array([])

    model = Net(seed, in_dim, out_dim, hid_dim)

    # TB summary
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    tensorboard_dir = os.path.join(DIR_PATH, 'logs', str(timestamp) + tag)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    params_init = hamiltorch.util.flatten(model).to(device).clone()
    print(params_init.shape)
    tau_list = []
    # tau = tau #10.  # ./100. # 1/50
    for w in model.parameters():
        #     print(w.nelement())
        #     tau_list.append(tau/w.nelement())
        tau_list.append(tau)
    tau_list = torch.tensor(tau_list).to(device)

    d = params_init.shape[0]

    prior = None
    previous_hmc_params = None # TODO: remove this when flow debug has finished

    n_tasks = 1 if one_task else data_gen.max_iter

    for task_id in range(n_tasks):

        print("\n\t Task {} \n".format(task_id + 1))

        plot_tag = tag + '_task_' + str(task_id + 1)
        # Plot visualisation in 2D for the toy dataset
        if toy_gaussians:
            x_train, y_train, x_test, y_test, x_test_full, y_test_full = data_gen.next_task()
            test_full_loader = DataLoader(
                TensorDataset(torch.Tensor(x_test_full), torch.Tensor(y_test_full).long()),
                len(x_test_full),
                shuffle=False)
        else:
            x_train, y_train, x_test, y_test, _, _ = data_gen.next_task()
        train_loader = DataLoader(TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long()), batch_size,
                                  shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long()), batch_size,
                                 shuffle=False)
        test_loaders.append(copy.deepcopy(test_loader))

        if load_hmc_params_tag is not None:
            print("Using cached HMC params")
            with open('results/hmc_samples_{0}_task{1}.pickle'.format(load_hmc_params_tag, task_id + 1), 'rb') as file:
                d = pickle.load(file)
                hmc_params = d['hmc_params']
            ess, acc_rate = np.nan, np.nan
            print("Cached HMC params of size: {}".format(hmc_params.shape))
            hmc_params = torch.from_numpy(hmc_params).float().to(device)
        else:
            hmc_params, ess, acc_rate = train(
                model,
                train_loader,
                params_init,
                tau_list,
                task_id,
                n_chains=n_chains,
                n_samples=n_samples,
                prior=prior,
                debug=debug,
                previous_hmc_params=previous_hmc_params,
            )  # hmc params is of size (n_chains, n_samples)
            ess_lst.append(ess)

        if cache_hmc_samples:
            with open('results/hmc_samples_{0}_task{1}.pickle'.format(tag, task_id + 1), 'wb') as file:
                pickle.dump({'hmc_params': hmc_params.cpu().numpy()}, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Access convergence
        _, _ = test(
            model,
            test_loader,
            hmc_params[0],  # only pass in one chain for eval
            tau_list,
            prior=prior,
            tag=str(task_id + 1),
            writer=writer,
        )

        # Test on all previous tasks
        accs, nlls = [], []
        for tl in test_loaders:
            acc, nll = test(
                model,
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

        writer.add_scalar("ess", ess, task_id + 1)
        writer.add_scalar("acc_rate", acc_rate, task_id + 1)
        writer.add_scalar("acc", acc[-1], task_id + 1)
        writer.add_scalar("nll", nll[-1], task_id + 1)

        # Plot visualisation in 2D for the toy dataset
        if toy_gaussians:
            plot_2d(model, hmc_params[-1], tau_list, prior, test_full_loader, data_gen, task_id, tag)

        # propagate posterior

        # Split dataset in a train and validation
        hmc_params_arr = hmc_params.cpu().numpy().reshape(-1, d)  # (n_chains, ..., d) --> (n_chains x ..., d)
        N = hmc_params_arr.shape[0]
        perm_inds = list(range(N))
        np.random.shuffle(perm_inds)  # shuffles in-place do not re-assign

        if mixture_of_gaussians:
            # Iterate over different number of components to the GMM and see which maximises the ll
            lls = []
            n_comps = [int(n) for n in np.geomspace(1, max_n_comps, 20)]

            for n in n_comps:
                posterior = MixtureOfGaussians(n_components=n, max_iter=500, n_init=5)
                posterior.fit(hmc_params_arr[perm_inds[:int(0.75 * N)], :])
                ll = posterior.goodness_of_fit(hmc_params_arr[perm_inds[int(0.75 * N):], :])
                lls.append(ll)
                writer.add_scalar("task" + str(task_id) + "mog_ll", ll, n)

            # Fit GMM to optimal number of components
            opt_n_comp = n_comps[np.argmax(lls)]
            print("Optimal number of components: {}".format(opt_n_comp))
            posterior = MixtureOfGaussians(n_components=opt_n_comp)
            posterior.fit(hmc_params_arr[perm_inds[:int(0.75 * n_samples)], :])
            posterior.sklearn_2_pytorch(d)
            writer.add_scalar("mog_opt_n_comps", opt_n_comp, task_id + 1)

            # generate samples from GMM and assess the accuracy of the BNN
            samples = posterior.sample(n_samples=32)
            samples = torch.from_numpy(samples[0]).float().to(device)
            acc, nll = test(
                model,
                test_loader,
                samples,
                tau_list,
                prior=prior,
                tag=plot_tag,
            )
            writer.add_scalar("gmm_sampled_acc", np.mean(acc.numpy()), task_id + 1)
            writer.add_scalar("gmm_sampled_nll", np.mean(nll.numpy()), task_id + 1)

        if flow:
            # define flow parameters
            d = params_init.shape[0]  # input dim, the number of parameters

            if mixture_of_gaussians:
                print("Using GMM as initialization to flow density estimator.")
                prior = posterior.gmm # Using the GMM as a prior for the flow
            else:
                prior = MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device))

            num_flows = n_flows
            h = flow_dim  # the number of neurons in scale (s) and translation (t) nets
            num_epochs = 200  # max. number of epochs

            # initialize RealNVP flow
            posterior = NormalizingFlow(
                seed=seed,
                num_flows=num_flows,
                prior=prior,
                d=d,
                h=h,
                lr=flow_lr,
                dequantization=False,
                decay_every=flow_decay_every,
                weight_decay=flow_weight_decay,
            ).to(device)

            # summary of the model
            print(summary(posterior, torch.zeros(1, d).to(device), show_input=True, show_hierarchical=False))

            training_loader = DataLoader(hmc_params_arr[perm_inds[:int(0.75 * N)], :], batch_size=32, shuffle=True)
            val_loader = DataLoader(hmc_params_arr[perm_inds[int(0.75 * N):], :], batch_size=32, shuffle=True)

            # fit flow parameters to the posterior samples from our HMC chain
            _ = flow_fit(
                model=posterior,
                task_id=task_id,
                num_epochs=num_epochs,
                training_loader=training_loader,
                val_loader=val_loader,
                writer=writer,
                name="{0}_task{1}".format(tag, task_id + 1),
                early_stopping=flow_early_stopping,
            )

            # generate samples from flow and assess the accuracy of the BNN
            samples = posterior.sample(batch_size=24)
            acc, nll = test(
                model,
                test_loader,
                samples,
                tau_list,
                prior=prior,
                tag=plot_tag,
            )
            writer.add_scalar("flow_sampled_acc", np.mean(acc.numpy()), task_id + 1)
            writer.add_scalar("flow_sampled_nll", np.mean(nll.numpy()), task_id + 1)


        # use posterior as prior for new task
        if flow or mixture_of_gaussians:
            prior = posterior
        else:
            prior = None

    writer.close()

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
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--one_task', dest='one_task', action='store_true', default=False)

    # Posterior propagation params
    parser.add_argument('--mixture_of_gaussians', dest='mixture_of_gaussians', action='store_true', default=False)
    parser.add_argument('--max_n_comps', type=int, default=100, help='maximum number of MoG components to search over.')
    parser.add_argument('--flow', dest='flow', action='store_true', default=False)
    parser.add_argument('--cache_hmc_samples', dest='cache_hmc_samples', action='store_true', default=False,
                        help='whether to cache the hmc samples to then tune the flow.')
    parser.add_argument('--n_flows', type=int, default=4, help='number of flows in the RealNVP.')
    parser.add_argument('--flow_dim', type=int, default=100, help='hidden state dim of the flow.')
    parser.add_argument('--flow_decay_every', type=int, default=5000, help='learning rate decay for flow.')
    parser.add_argument('--flow_early_stopping', dest='flow_early_stopping', action='store_true', default=False, help='Whether to train the flow with early stopping.')

    # BNN params
    parser.add_argument('--hid_dim', type=int, default=80)
    parser.add_argument('--tau', type=int, default=10, help='Gaussian prior precision.')

    # HMC params
    parser.add_argument('--n_chains', type=int, default=1)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=2500)

    args = parser.parse_args()
    params = vars(args)

    print(params)

    assert not (params['permuted'] and params['multiclass'])

    if params['dataset'] == 'permuted':
        n_tasks = 10
    else:
        n_tasks = 5
    one_task = params['one_task']
    hid_dim = params['hid_dim']
    tau = params['tau']
    runs = params['n_runs']
    n_chains = params['n_chains']
    accs, nlls = np.zeros((runs, n_tasks, n_tasks)), np.zeros((runs, n_tasks, n_tasks))
    ess_all = np.zeros((runs, n_tasks))
    toy_gaussians = False
    for i in range(runs):
        seed = params['seed'] + i
        if params['dataset'] == 'permuted':
            N = params['dataset_sz']
            data_gen = PermutedMnistGenerator(max_iter=n_tasks)
        elif params['dataset'] == 'toy_gaussians':
            N = params['dataset_sz']
            data_gen = ToyGaussiansGenerator(N=N, cl3=False, flatten_labels=True)
        elif params['dataset'] == 'toy_continuous_gaussians':
            N = params['dataset_sz']
            data_gen = ToyGaussiansContGenerator(num_samples=N, flatten_labels=True)
            toy_gaussians = True
        elif params['dataset'] == 'split_mnist':
            data_gen = SplitMnistGenerator(cl3=params['multiclass'], fashion=params['fashion'])
        else:
            raise ValueError
        acc, nll, ess = run(data_gen, seed, one_task, hid_dim, N, tau=tau, tag=params['tag'] + "_run_" + str(i),
                            n_chains=n_chains, n_samples=params['n_samples'],
                            mixture_of_gaussians=params['mixture_of_gaussians'], max_n_comps=params['max_n_comps'],
                            flow=params['flow'], cache_hmc_samples=params['cache_hmc_samples'],
                            n_flows=params['n_flows'], flow_dim=params['flow_dim'],
                            flow_decay_every=params['flow_decay_every'], flow_early_stopping=params['flow_early_stopping'],
                            debug=params['debug'], toy_gaussians=toy_gaussians)
        accs[i, :, :] = acc
        nlls[i, :, :] = nll
        ess_all[i, :] = ess
        print(acc)

    with open('results/hmc_res_{}.pickle'.format(params['tag']), 'wb') as handle:
        pickle.dump({'accs': accs, 'nlls': nlls, 'ess': ess_all}, handle, protocol=pickle.HIGHEST_PROTOCOL)
