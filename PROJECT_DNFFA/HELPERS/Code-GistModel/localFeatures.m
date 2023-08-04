function [vC, ig] = localFeatures(img, G, Nblocks);
% img must have same size than G

[n n Nfilters] = size(G);
img = img - mean(img(:));

I = fft2(img); 
IG = repmat(I,[1 1 Nfilters]).*G;
ig = ifft2(IG);
ig = abs(ig);

vC = downN(ig, Nblocks);

