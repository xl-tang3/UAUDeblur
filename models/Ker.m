
x = double(imread('im_1.png'))/255;  % clean image
SIZE = size(x);
% true = fspecial('motion', 20,10);  % true kernel
% inacu = fspecial('motion',20,20);  % inaccurate kernel
% true = fspecial('Gaussian',20,3);  
% inacu = fspecial('Gaussian',20,5);
true = fspecial('disk',7);
inacu = fspecial('disk',10);
PSFext = extendHforConv(true,SIZE(1),SIZE(2));
fPSFext = fft2(PSFext);
K1 = @(x) ifft2(fft2(x).*fPSFext);
y = K1(x);                          % true blurry

PSFext = extendHforConv(inacu,SIZE(1),SIZE(2));
fPSFext = fft2(PSFext);
K2 = @(x) ifft2(fft2(x).*fPSFext);
y1 = K2(x);                         
w_true = y - y1;
imshow(abs(w_true),[])
imwrite(y,'img1/im_1_ker_2.png')
imwrite(inacu, 'img1/k_2_im_1_levin.png')
PSF = fPSFext;
w = w_true;
save('K2.mat','PSF')
save('W2.mat','w')