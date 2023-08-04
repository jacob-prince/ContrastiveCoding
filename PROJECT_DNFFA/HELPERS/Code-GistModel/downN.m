function y=downN(x,N)
% averaging over non-overlapping spatial blocks

nx=fix(linspace(0,size(x,1),N+1));
ny=fix(linspace(0,size(x,2),N+1));
y = zeros(N, N, size(x,3));
for xx=1:N
  for yy=1:N
    v=mean(mean(x(nx(xx)+1:nx(xx+1), ny(yy)+1:ny(yy+1),:),1),2);
    y(xx,yy,:)=v(:);
  end
end
