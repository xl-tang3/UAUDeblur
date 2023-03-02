
x = double(imread('cameraman.png'))/255;  % clean image
SIZE = size(x);
 true = fspecial('motion', 20,10);  % true kernel
inacu = fspecial('motion',20,20);  % inaccurate kernel
% true = fspecial('Gaussian',20,4);  
% inacu = fspecial('Gaussian',20,4);
% true = fspecial('disk',5);
% inacu = fspecial('disk',5);
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
imwrite(y,'cameraman_blurry.png')
PSF = fPSFext;
w = w_true;
save('K.mat','PSF')
save('w.mat','w')