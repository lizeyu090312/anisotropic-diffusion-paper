# Variational Trajectory Optimization of Anisotropic Diffusion Schedules

Code for **"Variational Trajectory Optimization of Anisotropic Diffusion Schedules"** .  
We introduce a **trajectory-level framework** that learns matrix-valued diffusion schedules `M_t = g(t)V + h(t)(I - V)` jointly optimizing both the score network and the anisotropic noise allocation across subspaces. This repository builds on top of [NVLabs/EDM (Karras et al., 2022)](https://github.com/NVlabs/edm).

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

> **Note:**  
> The subdirectories `torch_utils/`, `training/`, and `dnnlib/`, as well as several utility scripts  
> (`train.py`, `generate.py`, `fid.py`, `dataset_tool.py`) are directly inherited from the  
> [NVLabs/EDM](https://github.com/NVlabs/edm) repository.

## Usage

### Dataset Preparation

Our dataset preparation follows exactly the same procedure as in the official [NVLabs/EDM Preparing datasets](https://github.com/NVlabs/edm).

1. Download the original datasets (e.g., CIFAR-10, AFHQv2, FFHQ) from their official websites.

2. Convert datasets to the desired resolution. Use the EDM utility `dataset_tool.py` to process the downloaded datasets into the resolution required by our experiments. For example:
   ```bash
   # Convert CIFAR-10 images to 32×32
   python dataset_tool.py --source=<path_to_cifar10_folder> --dest=datasets/cifar10-32x32.zip --resolution=32

   # Convert AFHQv2 images to 64×64
   python dataset_tool.py --source=<path_to_afhqv2_folder> --dest=datasets/afhqv2-64x64.zip --resolution=64

   # Convert FFHQ images to 64×64
   python dataset_tool.py --source=<path_to_ffhq_folder> --dest=datasets/ffhq-64x64.zip --resolution=64
   ```

3. Ensure dataset directory matches training scripts. The training scripts expect the processed datasets to be located under:
   ```
   datasets/
     ├── cifar10-32x32.zip
     ├── afhqv2-64x64.zip
     └── ffhq-64x64.zip
   ```

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
