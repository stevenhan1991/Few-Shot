# A Few-Shot Learning Framework for Time-Varying Scientific Data Generation via Conditional Diffusion Model
Pytorch implementation for A Few-Shot Learning Framework for Time-Varying Scientific Data Generation via Conditional Diffusion Model

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.7
- Numpy
- Skimage
- Pytorch >= 1.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train' --dataset 'Vortex' --applicaion 'temporal'
```

- inference
```
python3 main.py --mode 'inf' --dataset 'Vortex' --application 'temporal'
```
  year={2023}
}

