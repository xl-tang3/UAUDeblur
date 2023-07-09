# Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior
This is the official `Python` implementation of the [CVPR 2023](https://cvpr.thecvf.com/)  paper **Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior**
The repository contains reproducible `PyTorch` source code for computing the deblurred image and residual given a single kernel and blurry image.

Here are the [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Uncertainty-Aware_Unsupervised_Image_Deblurring_With_Deep_Residual_Prior_CVPR_2023_paper.pdf) and [Supplementary material](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Tang_Uncertainty-Aware_Unsupervised_Image_CVPR_2023_supplemental.pdf).
## Citation
```
@InProceedings{Tang_2023_CVPR,
    author    = {Tang, Xiaole and Zhao, Xile and Liu, Jun and Wang, Jianli and Miao, Yuchun and Zeng, Tieyong},
    title     = {Uncertainty-Aware Unsupervised Image Deblurring With Deep Residual Prior},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9883-9892}
}
```
<p align="center"><img src="pics/Model.png" width="700" /></p>

## Visualization of the kernel induced error (residual)
<p align="center"><img src="pics/res.png" width="500" /></p>

## Test on real blurry images from Lai dataset
<p align="center"><img src="pics/real.png" width="400" /></p>

## Robustness to the kernel error
<p align="center"><img src="pics/robustness.png" width="400" /></p>

If the inputs contain only blurry images, you will need to run some outsourcing kernel estimation algorithm to obtain the kernel. The datasets folder include some estimated kernel and corresponding blurry images for test.

Before runing `UDeblur_demo.py`, the first thing you have to do is to pre-process the (estimated) kernel with Fourier transformation and save it as a `.mat` file. We provide an example for this procdure, see `KernelGen_Inaccu.m`; where the FFT of blur kernel is done via the function `extendHforConv.m`.

If you have any problem, contact me at Sherlock315@163.com

