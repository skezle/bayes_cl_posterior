import os
import sys
sys.path.append('../../')

import argparse
import numpy as np
import torch

from protocl.main.protocl import ProtoCL
from data import SplitMnistGenerator, MnistGenerator, SplitCIFAR100Generator, SplitCIFAR10Generator
from protocl.experiments.api import run


# directory for the current file
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')

    # data args
    parser.add_argument('--data.dataset', type=str, default='SplitMNIST', metavar='DS',
                        help="data set name (default: {:s})".format('SplitMNIST'))
    parser.add_argument('--data.batch_size', type=int, default=50, metavar='BATCHSIZE',
                        help="meta batch size (default: 50)")
    parser.add_argument('--data.cuda', type=int, default=-1, metavar='CUDA_DEVICE',
                        help='which cuda device to use. if -1, uses cpu')

    # model args
    parser.add_argument('--model.model', type=str, default='main', metavar='model',
                        help="which ablation to use (default: main moca model)")
    parser.add_argument('--model.x_dim', type=int, default=1, metavar='XDIM',
                        help="dimensionality of input images (default: '1,28,28')")
    parser.add_argument('--model.hid_dim', type=int, default=128, metavar='HIDDIM',
                        help="dimensionality of hidden layers in encoder (default: 64)")
    parser.add_argument('--model.y_dim', type=int, default=1, metavar='YDIM',
                        help="number of classes/dimension of regression label")
    parser.add_argument('--model.phi_dim', type=int, default=32, metavar='PDIM',
                        help="dimensionality of embedding space (default: 64)")
    parser.add_argument('--model.sigma_eps', type=str, default='[0.05]', metavar='SigEps',
                        help="noise covariance (regression models; Default: 0.05)")
    parser.add_argument('--model.Linv_init', type=float, default=0., metavar='Linv',
                        help="initialization of logLinv in ProtoCL (Default: 0.0)")
    parser.add_argument('--model.dirichlet_scale', type=float, default=10., metavar='Linv',
                        help="value of log Dirichlet concentration params (init if learnable; Default: 0.0)")
    parser.add_argument('--model.large_model', type=bool, default=False,
                        help='whether to use a resnet12 backbone.')
    parser.add_argument('--model.data_aug', type=bool, default=False,
                        help='whether to use data_aug.')

    # train args
    parser.add_argument('--train.n_epochs', type=int, default=100, metavar='NEPOCHS',
                       help='number of episodes to train (default: 100)')
    parser.add_argument('--train.learning_rate', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.02)')
    parser.add_argument('--train.weight_decay', type=float, default=0.00, metavar='WD',
                        help='optim weight decay (default: 0.00)')
    parser.add_argument('--train.decay_every', type=int, default=1500, metavar='LRDECAY',
                        help='number of iterations after which to decay the learning rate')
    parser.add_argument('--train.dropout_prob', type=float, default=0.0, metavar='DROPOUT',
                        help='dropout prob.')
    parser.add_argument('--train.learnable_noise', type=int, default=0, metavar='learn_noise',
                        help='enable noise being learnable (default: false/0)')
    parser.add_argument('--train.learnable_dirichlet', type=int, default=0, metavar='learn_dir',
                        help='enable dirichlet concentration being learnable')
    parser.add_argument('--train.verbose', type=bool, default=True, metavar='verbose',
                        help='print during training (default: True)')
    parser.add_argument('--train.seed', type=int, default=1, metavar='SEED',
                        help='numpy seed')
    parser.add_argument('--train.experiment_id', type=int, default=0, metavar='SEED',
                        help='unique experiment identifier seed')
    parser.add_argument('--train.experiment_name', type=str, default=0, metavar='SEED',
                        help='name of experiment')
    parser.add_argument('--train.val_iterations', type=int, default=5, metavar='NEPOCHS',
                        help='number of episodes to validate on (default: 5)')
    parser.add_argument('--train.debug_grads', type=int, default=0, metavar='debug_grads',
                        help='enable debugging of gradients on TB (default: false/0)')
    parser.add_argument('--train.val', type=bool, default=False, metavar='val',
                        help='whether to evaluate on validation set and not the test set.')

    # experience replay args
    parser.add_argument('--memory.memory_size', type=int, default=1000, metavar='MEM_SIZE',
                        help='Size of the memory per task.')

    config = vars(parser.parse_args())

    print(config)

    path = str(config['train.experiment_id']) + '_' + config['train.experiment_name'] + '_'
    path += config['data.dataset'] + '_'
    path += config['model.model'] + '_'
    path += str(config['train.seed']) + '_'

    for bool_arg in [
        'train.learnable_noise',
        'train.learnable_dirichlet',
        'train.debug_grads',
        'train.val',
        'model.large_model',
        'model.data_aug',
    ]:
        config[bool_arg] = bool(config[bool_arg])

    seed = config['train.seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(os.path.join(DIR_PATH, 'saved_models/')):
        os.makedirs(os.path.join(DIR_PATH, 'saved_models/'), exist_ok=True)

    # setup the dataset
    if config['data.dataset'] == 'SplitMNIST':
        data_gen = SplitMnistGenerator(cl3=True, fashion=False, one_hot=True)
        n_tasks = 5
    elif config['data.dataset'] == 'SplitFMNIST':
        data_gen = SplitMnistGenerator(cl3=True, fashion=True, one_hot=True)
        n_tasks = 5
    elif config['data.dataset'] == 'SplitCIFAR100':
        data_gen = SplitCIFAR100Generator(
            cl3=True, 
            data_aug=config["model.data_aug"],
        )
        n_tasks = 10
    elif config['data.dataset'] == 'SplitCIFAR10':
        data_gen = SplitCIFAR10Generator(
            cl3=True,
            val=True,
            data_aug=config["model.data_aug"],
        )
        n_tasks = 5
    elif config['data.dataset'] == 'MNIST':
        data_gen = MnistGenerator(fashion=False)
        n_tasks = 1
    else:
        raise NotImplementedError

    n_epochs = config['train.n_epochs']
    batch_size = config['data.batch_size']
    model = ProtoCL(config)

    run(
        model, 
        config,
        n_tasks, 
        data_gen,
        n_epochs, 
        batch_size,
        path,
        seed,
    )

    print("Finished training")