# A Few-Shot Learning Framework for Time-Varying Scientific Data Generation via Conditional Diffusion Model
Pytorch implementation for A Few-Shot Learning Framework for Time-Varying Scientific Data Generation via Conditional Diffusion Model

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.7
- Numpy
- Skimage
- Pytorch >= 2.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Training models
```
cd Code 
```

- training
```
python3 main.py --config_file vortex.yaml --device 0 --mode train --num_samples 3
```
where config_file stores all configurations for diffusion model inclduing model and training configurations, device indiates which GPU is used for training (the code only supports single GPU training), mode demonstrate whether we need to train or infer diffusion model, and num_samples determinates how many samples are used for training. 

The format of the config file can be found at configs folder. We have uploaded four .yaml files for vortex, ionization (PD), combustion (MF), and argonne bubble for reference.


- inference
```
python3 main.py  --confile_file vortex.yaml --device 0 --mode inf --num_samples 3
```

