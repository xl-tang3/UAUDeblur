% Change the file source correspondingly to make it work
y = double(imread('.\datasets\ker_levin\im_4_ker_8.png'))/255;   % true blurry
y1 = y(:,:,1);
imwrite(y1,'blurry.png');
y = y1;
x = double(imread('.\datasets\sharp\im_4.png'))/255;
imwrite(x(:,:,1),'clean.png')
SIZE = size(x(:,:,1));
% k = double(imread('.\datasets\Levin\BD_cho_and_lee_tog_2009\k_8_im_4_cho_and_lee.png'));
k = double(imread('.\datasets\ker_levin\BD_sun_iccp_2013\kernel_estimates\k_8_im_4_sun_2013.png'));
inacu = k/sum(k(:));
PSFext = extendHforConv(inacu,SIZE(1),SIZE(2));
fPSFext = fft2(PSFext);
PSF = fPSFext;
K1 = @(x) ifft2(fft2(x).*fPSFext);
y1 = K1(x(:,:,1));
w = y-y1;
save('w1.mat','w')
save('K1.mat','PSF')
