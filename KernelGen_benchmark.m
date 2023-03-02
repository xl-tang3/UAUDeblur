% Change the file source correspondingly to make it work
y = double(imread('.\Datasets\Lai\ker_lai\natural_02_kernel_01.png'))/255;   % true blurry
y1 = y(:,:,1);
imwrite(y1,'blurry.png');
y = y1;
x = double(imread('.\Datasets\Lai\sharp\natural_02.png'))/255;
imwrite(x(:,:,1),'clean.png')
SIZE = size(x(:,:,1));
% k = double(imread('D:\Pycharm\DRP\data\Levin_NK\BD_cho_and_lee_tog_2009\kernel_estimates\k_2_im_1_cho_and_lee.png'));
k = double(imread('.\Datasets\Lai\BD_perrone_cvpr_2014\kernel_estimates\psf_natural_02_kernel_01_14_perrone.png'))/255;
k = k(:,:,1);
inacu = k/sum(k(:));
PSFext = extendHforConv(inacu,SIZE(1),SIZE(2));
fPSFext = fft2(PSFext);
PSF = fPSFext;
K1 = @(x) ifft2(fft2(x).*fPSFext);
y1 = K1(x(:,:,1));
w = y-y1;
save('w1.mat','w')
save('K1.mat','PSF')
