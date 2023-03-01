% Generate the inaccurate kernel in frequency as input \hat k
x = double(imread('peppersRGB.png'))/255;  % clean image
SIZE = size(x);
true = fspecial('motion', 20,10);  % true kernel
inacu = fspecial('motion',35,10);  % inaccurate kernel
% true = fspecial('Gaussian',20,4);  
% inacu = fspecial('Gaussian',20,4);
% true = fspecial('disk',5);
% inacu = fspecial('disk',5);
PSFext = extendHforConv(inacu,SIZE(1),SIZE(2));
fPSFext = fft2(PSFext);                  
PSF = fPSFext;
save('K.mat','PSF')