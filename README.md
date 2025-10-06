# Trajectory-Optimal Anisotropic Diffusion Models

Code for **"Trajectory-Optimal Anisotropic Diffusion Models"** (ICLR 2026 submission 20950).  
We introduce a **trajectory-level framework** that learns matrix-valued diffusion schedules  
\[
M_t = g(t)V + h(t)(I - V),
\]
jointly optimizing both the score network and the anisotropic noise allocation across subspaces.

## Repository Structure
```
anisotropic-diffusion-paper/
├── common_utils.py                                 # Core math utilities and diffusion operators
├── g_iso_train.py                                  # Train isotropic schedule
├── g_iso_train_discretize.py                       # Train isotropic schedule minimizing discretization error
├── g_ani_train.py                                  # Train anisotropic schedule
├── g_iso_wrapper_train.py                          # Train isotropic training with wrapper
├── g_ani_wrapper_train.py                          # Train anisotropic training with wrapper
├── g_sample.py                                     # Unified sampling interface
├── run_all_train_and_sample.sh                     # Shell script for all datasets
├── requirements.txt                                # Environment dependencies
├── torch_utils/                                    # From NVLabs/edm
├── training/                                       # From NVLabs/edm
├── dnnlib/                                         # From NVLabs/edm
└── train.py, generate.py, fid.py, dataset_tool.py  # From NVLabs/edm
```

## Usage

### Training
```bash
# Train isotropic schedule (CIFAR-10 example)
python g_iso_train.py --dataset cifar10

# Train anisotropic schedule
python g_ani_train.py --dataset afhqv2

# Train wrapper version
python g_iso_wrapper_train.py --dataset cifar10
python g_ani_wrapper_train.py --dataset afhqv2
```

### Sampling
```bash
# Sample images
python g_sample.py --dataset cifar10 --model_tag g-iso --steps 20

# Example for anisotropic model
python g_sample.py --dataset afhqv2 --model_tag g-ani --steps 20
```


## Environment
```bash
conda create -n anidiff python=3.9
conda activate anidiff
pip install -r requirements.txt
```
