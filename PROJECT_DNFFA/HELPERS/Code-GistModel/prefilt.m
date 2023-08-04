function [output]=prefilt(input,fc)
% ima = prefilt(ima);
% fc=4 works ok

if min(input(:))<0 error('NEGATIVE INPUT'); end
if nargin==1
    fc=8;
end

input=log(input/max(input(:))*254+1);
n=size(input,1); % taille
im=input(8:n-8,8:n-8);
input=(input-mean(input(:)))/std(im(:));

% eliminer le probleme de gamma

% enleve les effets d'ombres
% estimation de l'ecart type, localement sur un patch de taille m x m
% fc1=8;
fc1=fc; % C'est moins bruite

s1=fc1/sqrt(log(2));
[fx,fy]=meshgrid(0:n-1);fx=fx-n/2;fy=fy-n/2;
gf1=fftshift(exp(-(fx.^2+fy.^2)/n^2*256^2/(s1^2)));
pb=real(ifft2(fft2(input).*gf1));
ph=input-pb;
imanorm=sqrt(real(ifft2(fft2(ph.^2).*gf1))); 

% normalisation de l'ecart type
output=ph./(.2+imanorm);

%output=output.^2./(std(output(:).^2)+output.^2);

% centre reduction de l'image
%output=(output-mean(output(:)))/std(output(:));

if (1==0)
   [x,y]=meshgrid(1:256,1:256);
   
   xm=sum(sum(x.*imanorm.^2))/sum(sum(imanorm.^2))
   ym=sum(sum(x.*imanorm.^2))/sum(sum(imanorm.^2))
   
   w=1-exp(-.0001*((x-xm).^2+(y-ym).^2));
   output=output.*w;
end
