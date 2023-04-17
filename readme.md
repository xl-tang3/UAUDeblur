# Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior
This is the official `Python` implementation of the [CVPR 2023](https://cvpr.thecvf.com/)  paper **Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior**

The repository contains reproducible `PyTorch` source code for computing the deblurred image and residual given a single kernel and blurry image.

<p align="center"><img src="pics/Model.png" width="700" /></p>

## Visualization of the kernel induced error (residual)
<p align="center"><img src="pics/res.png" width="500" /></p>

## Test on real blurry images from Lai dataset
<p align="center"><img src="pics/real.png" width="400" /></p>

## Robustness to the kernel error
<p align="center"><img src="pics/robustness.png" width="400" /></p>



If the inputs contain only blurry images, you will need to run some kernel estimation algorithm to obtain the kernel.
