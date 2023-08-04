function G = createRosasGabor(numberOfOrientationsPerScale, imageSize)
% G = transfer functions for a jet of gabor filters
% numberOfOrientationsPerScale = vector that contains the number of
%                                orienations at each scale
% imageSize = 

if ~exist('imageSize')
    imageSize = 256;
end

% number of orientations for each scale (from HF to BF)
if ~exist('numberOfOrientationsPerScale')
    or = [12 8 6 6 6 4]; % ces numbres doivent etre paires
else
    or = numberOfOrientationsPerScale;
end

Nscales = length(or);
Nfilters = sum(or);

l=0;
for i=1:Nscales
    for j=1:or(i)
        l=l+1;
        %param = [sigma_r, fr, sigma_theta, theta]
        param(l,:)=[.35 .3/(1.85^(i-1)) 16*or(i)^2/32^2 pi/(or(i))*(j-1)];
    end
end
H=gabor_set(param, imageSize);

% Coupes a -3dB pour les filtres.
G = zeros([imageSize imageSize Nfilters]);
for i=1:Nfilters
    G(:,:,i)=fftshift(reshape(H(:,i),imageSize,imageSize));
end



% figure
% for i=1:Nfilters
%     max(max(G(:,:,i)))
%     contour(fftshift(G(:,:,i)),[1 .7],'r');
%     hold on
%     drawnow
% end
% axis('on')
% axis('square')
% axis('ij')


