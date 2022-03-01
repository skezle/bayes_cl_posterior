# Can Sequential Bayesian Inference Solve Continual Learning

This code accompanies the corresponding AABI 2022 paper, [here](https://openreview.net/forum?id=2Ann7eaLBEv).

## Setup

We use the HMC package `hamiltorch` to perform HMC sampling. We have added some custom functions, so please use the implementation in this repo.

## Running HMC for CL

This will use the GMM as a density estimator over HMC samples as a prior for a new task:

```python run_hmc_cl.py --seed=0 --dataset=toy_gaussians --mixture_of_gaussians --hid_dim=10 --n_chains=20 --n_runs=1 --n_samples=5000 --max_n_comps=500 --tag=mog_tg_s0``` 

## Citation

```
@inproceedings{kessler2021can,
  title={Can Sequential Bayesian Inference Solve Continual Learning?},
  author={Kessler, Samuel and Cobb, Adam D and Zohren, Stefan and Roberts, Stephen J},
  booktitle={Fourth Symposium on Advances in Approximate Bayesian Inference},
  year={2021}
}
```

## Acknowledgements

Thanks to the maintainers of the [hamiltorch](https://github.com/AdamCobb/hamiltorch) and the [MOCA](https://github.com/StanfordASL/moca) paper from which the implementation of PCOC I use for Prototypical Bayesian CL.




 
