# Noise-Inspired Diffusion Model for Generalizable Low-Dose CT Reconstruction
This is the official implementation of the paper "Noise-Inspired Diffusion Model for Generalizable Low-Dose CT Reconstruction".

## Updates
- Dec, 2024: initial commit.


## Data Preparation
- The AAPM-Mayo dataset can be found from: [Mayo 2016](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/). 
- The "Low Dose CT Image and Projection Data" can be found from [Mayo 2020](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026#527580262a84e4aa87794b6583c78dccf041269f).

## Training & Inference

## Requirements
```
- Linux Platform
- python==3.8.13
- cuda==10.2
- torch==1.10.1
- torchvision=0.11.2
- numpy=1.23.1
- scipy==1.10.1
- h5py=3.7.0
- pydicom=2.3.1
- natsort=8.2.0
- scikit-image=0.21.0
- einops=0.4.1
- tqdm=4.64.1
- wandb=0.13.3
```

## Acknowledge
- Our codebase builds heavily on [DU-GAN](https://github.com/Hzzone/DU-GAN) and [Cold Diffusion](https://github.com/arpitbansal297/Cold-Diffusion-Models). Thanks for open-sourcing!
```
