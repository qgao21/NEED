# Noise-Inspired Diffusion Model for Generalizable Low-Dose CT Reconstruction
This is the official implementation of the paper "Noise-Inspired Diffusion Model for Generalizable Low-Dose CT Reconstruction".

## Updates
- Dec, 2024: initial commit.

## Data Preparation
- The AAPM-Mayo dataset can be found from: [Mayo 2016](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/). 
- The "Low Dose CT Image and Projection Data" can be found from [Mayo 2020](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026#527580262a84e4aa87794b6583c78dccf041269f).

## Requirements
```
- Linux Platform
- python==3.8.13
- cuda==10.2
- torch==1.10.1
- torchvision=0.11.2
- numpy=1.23.1
- scipy==1.10.1
- pydicom=2.3.1
- natsort=8.2.0
- scikit-image=0.21.0
- einops=0.4.1
- tqdm=4.64.1
- wandb=0.13.3
```

## Acknowledge
- Our codebase builds heavily on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [Cold Diffusion](https://github.com/arpitbansal297/Cold-Diffusion-Models), and, [DU-GAN](https://github.com/Hzzone/DU-GAN). Thanks for open-sourcing!
- We use the [ASTRA Toolbox](https://astra-toolbox.com/) for low-dose CT data simulation and leverage the [TorchRadon Toolbox](https://github.com/matteo-ronchetti/torch-radon) for fast FBP reconstruction. Thank you for open-sourcing these valuable tools!
