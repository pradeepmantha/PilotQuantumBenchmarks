# Ensure all necessary modules are loaded up-front

module load PrgEnv-gnu/8.3.3 cray-mpich/8.1.25 cudatoolkit/11.7 craype-accel-nvidia80 evp-patch gcc/11.2.0

# Required due to a potentially missing lib for the CUDA-aware MPICH library

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH

# Create a python virtualenv

python -m venv pyenv

# install the following dependencies

python -m pip install cmake ninja custatevec-cu11 wheel pennylane~=0.32.0 pennylane-lightning~=0.32.0

# build mpi4py against the system's CUDA-aware MPICH

MPICC="cc -shared" python -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

# clone and checkout lightning.gpu at version 0.32.0

git clone https://github.com/PennyLaneAI/pennylane-lightning-gpu

cd pennylane-lightning-gpu && git checkout v0.32.0

# build the extension module with MPI support, and package it all into a wheel

python setup.py build_ext --define="PLLGPU_ENABLE_MPI=ON;-DCMAKE_CXX_COMPILERS=$(which mpicxx);-DCMAKE_C_COMPILER=$(which mpicc)"

python setup.py bdist_wheel

# Assuming the above steps are reproduced, your wheel can be installed as needed

python -m pip install ./dist/*.whl

# Grab an allocation for however many GPUs you want; I'd start with a single interactive node/4 GPUs

salloc -N 1 --qos interactive  --time 02:00:00 --gpus=4 --ntasks-per-node=4 --gpus-per-task=1 --constraint gpu --account=

# and launch over multiple processes

srun -n 4 python script.py

