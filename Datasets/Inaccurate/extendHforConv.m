function [ConvH] = extendHforConv(H, ny, nx)
    ConvH = zeros(ny,nx);
    [m2y,m2x]=size(H);
    my = floor(m2y/2);
    mx = floor(m2x/2); %centre is mx+1
    ConvH(1:m2y,1:m2x) = H;
    ConvH = circshift(ConvH, [-my -mx]);
return;
