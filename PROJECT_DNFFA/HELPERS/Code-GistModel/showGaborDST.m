function [DST]=showGaborDST(G,dst,pc,N);
% [DST]=showDSTlocal(H,dst,ni,N);
%

dst = pc*dst(1:end-1);

[ni ni Nfilters] = size(G);
for i=1:Nfilters
    G(:,:,i) = fftshift(G(:,:,i));
end
H = reshape(G, [ni*ni Nfilters]);
n = size(H,2);

dst = reshape(dst, [N N n]);

% Calcul du DST local...
l=0;
Dst=zeros((ni+2)*N,(ni+2)*N);
win=zeros((ni+2)*N,(ni+2)*N);
for ly=1:N; 
   for lx=1:N
      l=l+1;
      %DST=reshape(H*dst(1+(l-1)*n:l*n),ni,ni); 
      DST=reshape(H*squeeze(dst(lx, ly, :)),ni,ni); 
      DST(2:ni,2:ni)=DST(2:ni,2:ni)+DST(ni:-1:2,ni:-1:2);
      if N==1
          
         DST(:,2:ni)=DST(:,2:ni)+DST(:,ni:-1:2); % Mirror
      end
      Dst((ni+2)*(lx-1)+1:(ni+2)*(lx-1)+ni,(ni+2)*(ly-1)+1:(ni+2)*(ly-1)+ni)=DST;
      win((ni+2)*(lx-1)+1:(ni+2)*(lx-1)+ni,(ni+2)*(ly-1)+1:(ni+2)*(ly-1)+ni)=1+0*DST;
   end
end
if N>1 
   DST=Dst;
end
 
if nargout==0
    figure
   image((Dst/max(abs(Dst(:)))*128+128).*win+256*(1-win))
   %image(Dst/max(abs(Dst(:))))
   axis('normal')
   axis('off')
   colormap(gray(256))
end


