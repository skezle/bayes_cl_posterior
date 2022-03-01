mkdir -p plots
mkdir -p data
mkdir -p results

# the file hamiltorch/hamiltorch/samplers.py has a new HMC sampler with a custom prior
#git clone git@github.com:AdamCobb/hamiltorch.git
cd hamiltorch
pip install -e .
