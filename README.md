# On Sequential Bayesian Inference for Continual Learning

This code accompanies the corresponding the paper https://arxiv.org/abs/2301.01828.

## Setup

We use the HMC package `hamiltorch` to perform HMC sampling. We have added some custom functions, so please use the implementation in this repo and run the script `setup.sh`.

We use a conda virtual environment for the experiments with HMC and for the ProtoCL implmentation. Please refer to `requirements.txt` for the package requirements.

## Running HMC for CL

To run the continual learning experiments over the toy 2-d dataset (Fig. 3) from  the paper using HMC with a GMM density estimator over samples:

`python main.py --seed=0 --dataset=toy_cont_gaussians --mixture_of_gaussians --hid_dim=10 --n_chains=20 --n_runs=1 --n_samples=5000 --max_n_comps=500 --tag=mog_tg_s0`

To run the continual learning experiments over the toy 2-d dataset (Fig. 3) from the paper using HMC with normalizing flow prior over HMC samples:

`python main.py --seed=0 --dataset=toy_continuous_gaussians --flow --hid_dim=10 --n_chains=10 --n_runs=1 --n_samples=5000 --tag=flow_tcg_test`

## Prototypical Bayesian Continual Learning

For Split MNIST:

`python main.py --data.cuda=0 --data.dataset=SplitMNIST --train.experiment_id=0 --train.seed=$seed --train.n_epochs=50 --train.experiment_name=SMNIST --model.x_dim=784 --model.y_dim=10 --model.phi_dim=128 --data.batch_size=64 --model.hid_dim=256 --model.dirichlet_scale=0.78 --train.learnable_dirichlet=1 --train.val=False --memory.memory_size=1000 --train.learning_rate=0.0001 --train.dropout_prob=0.1 --train.weight_decay=0.0001`

For Split fashion-MNIST:

`python main.py --data.cuda=0 --data.dataset=SplitFMNIST --train.experiment_id=0 --train.seed=0 --train.n_epochs=50 --train.experiment_name=SFMNIST --model.x_dim=784 --model.y_dim=10 --model.phi_dim=128 --data.batch_size=128 --model.hid_dim=256 --model.dirichlet_scale=0.78 --train.learnable_dirichlet=1 --train.val=False --memory.memory_size=1000 --train.learning_rate=0.0001 --train.dropout_prob=0.0 --train.weight_decay=0.01`

For Split CIFAR10:

`python main.py --data.cuda=0 --data.dataset=SplitCIFAR10 --train.experiment_id=0 --train.seed=0 --train.n_epochs=50 --train.experiment_name=scifar10 --model.y_dim=10 --data.batch_size=128 --model.hid_dim=200 --model.dirichlet_scale=0.78 --memory.memory_size=1000 --model.phi_dim=32 --train.learnable_noise=0 --train.learnable_dirichlet=0 --train.learning_rate=0.001`
   
For Split CIFAR100:

`python main.py --data.cuda=0 --data.dataset=SplitCIFAR100 --train.experiment_id=0 --train.seed=0 --train.n_epochs=200 --train.experiment_name=SCIFAR100 --model.phi_dim=250 --data.batch_size=32 --model.hid_dim=256 --model.dirichlet_scale=0.78 --trainlearnable_dirichlet=1 --memory.memory_size=1000 --train.learning_rate=0.001 --model.y_dim=10`

## Acknowledgements

Thanks to the maintainers of the [hamiltorch](https://github.com/AdamCobb/hamiltorch) and the [MOCA](https://github.com/StanfordASL/moca) paper from which the implementation of PCOC I use for Prototypical Bayesian CL.




 
