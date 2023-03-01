# Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior
This is the official `Python` implementation of the [CVPR 2023](CVPR.thecvf.com)  paper **Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior**

The repository contains reproducible `PyTorch` source code for computing the estimate the deblurred image and residual given a single kernel and blurry image.

<p align="center"><img src="pics/Model.png" width="400" /></p>
<p align="center"><img src="pics/real.png" width="400" /></p>
<p align="center"><img src="pics/robustness.png" width="400" /></p>


Run the m file 'KernelGen_benchmark/Inacu.m' for data prepraration.
This will transform the kernel input into FFT domain for the sake of computation and save all the inputs in 'mat' files.

If the inputs contain only blurry images, you will need to run some kernel estimation algorithm to obtain the kernel.
## For the test of benchmark datasets (Levin & Lai$)
Once the data inputs are prepared, you can run .py files to deblur whatever you want.
### Levin dataset
Run 'DIDRP_gray_demo.py'
### Inaccurate-kernel datasets
Run 'KernelGen_Inacu.m' to generate inaccurate kernels. For a motion blur with lengths of 20 pixels and orientations of 10 degrees:

`ker = fspecial('motion', 20, 10);`

Then run the .py files to obtain the deblurred rusults.
