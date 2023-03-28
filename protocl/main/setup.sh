mkdir -p src/plots
mkdir -p src/results
mkdir -p protocl/experiments/data
mkdir -p protocl/experiments/results
mkdir -p protocl/experiments/plots
mkdir -p data

# the file hamiltorch/hamiltorch/samplers.py has a new HMC sampler with a custom prior
#git clone git@github.com:AdamCobb/hamiltorch.git
cd hamiltorch
pip install -e .
