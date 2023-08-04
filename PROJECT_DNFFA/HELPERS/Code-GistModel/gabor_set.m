function [H,s,Ener]=gabor_set(param,in);
% function [H,s,Ener]=gabor_set(param,in);
%
% Programme pour calculer des filtres de Gabor 2D.
%
% ENTREES:
% param contiendra les parametres de tous les filtres de Gabor que
%       on veut generer:
%
%       param = [sigma_r, fr, sigma_theta, theta; ...]
%
% in    Image en entree ou taille d'image si on ne veut pas une image precise
%
% SORTIES:
% H     Fonctions de transfert analytiques des N filtres
% s     Sorties des N filtres rangees par colonnes
% Ener  Energies moyennes sur toute l'image pour les N filtres.
%
%
% COMENTAIRES D'UTILISATION:
% Le programe peut etre utilise sur differentes configurations car il 
% est sensible au nombre des variables utilisees en sortie.
%
% 1) On veut filtrer une image par une baterie de filtres de gabor:
%
%    [H,s,Ener]=gabor_set(param,image);
%    * H: Donne les fonctions de transfer des filtres utilises.
%    * s: Donne les images de sortie. La partie reel es la partie cosinus
%    et la partie imaginaire est la partie sinus. s est une matrice
%    de n*n lignes et N colognes. Pour recuperer une image il faut faire
%
%    imageout=reshape(s(:,i),n,n);
%    
%    Ceci vous donne l'image (complexe) en sortie du filtre i.
%
% 2) On veut generer unquement les fonctions de transfert pour le calcul rapide
%
%    [H]=gabor_set(param,n);
%    * H: Donne les fonctions de transfer des filtres utilisees.
%    * n: Tailles des images que seront utilisees.
%    
%    On peut ensuite utiliser le resultat H du facon suivante:
%    
%    S=fftshift(abs(fft2(ima)));
%    E=H'*S(:);  
%    
%    E contiendra les energies en sortie pour l'image 'ima' de toute la baterie
%    de filtres de Gabor.
%

%      Antonio Torralba, 19 Janvier 1999.
%      torralba@tirf.inpg.fr
%      http://www.tirf.inpg.fr/PERSON/torralba/torralba.html


  [N,m]=size(param);
  [n,n]=size(in);
  % Detecter si il y a une image en entree:
  if (n==1) n=in; end;
  
  % grille de frequences:
  [fx,fy]=meshgrid(0:n-1,0:n-1);
  fx=fx-n/2; fy=fy-n/2;
  
  % Coordonees polaires:
  fr=sqrt(fx.^2+fy.^2);
  t=angle(fx+j*fy);  

  % Transformee de Fourier
  IN=fftshift(fft2(in));
  
  % Calcul de filtres de Gabor:
  H=zeros(n*n,N);
  for i=1:N
      %  Filtre numero i:
      par=param(i,:);
      tr=t+par(4); tr=tr+2*pi*(tr<-pi)-2*pi*(tr>pi);

      %  Generation de la fonction de transfert analytique:
      if (par(2)>0)
         G=exp(-10*par(1)*(fr/n/par(2)-1).^2-2*par(3)*pi*tr.^2);
      else 
         G=exp(-10*par(1)*fr.^2-2*par(3)*pi*tr.^2);
      end
      
      H(:,i)=G(:);
      
      %  Si il y a au moins deux variables en sortie alors on calcule 
      % les sorties des filtres (complexes):
      if (nargout>1)
         S=(ifft2(fftshift(IN.*G)));
         s(:,i)=S(:);
      end
      
      %  Si il y a trois arguments en sortie, on calcule aussi les energies locales.
      if (nargout>2)
         Ener(i)=mean(abs(S(:)).^2);
      end
  end
  
