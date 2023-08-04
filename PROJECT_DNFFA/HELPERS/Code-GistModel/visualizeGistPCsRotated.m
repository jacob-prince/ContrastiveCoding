function fh = visualizeGistPCsRotated(pc, G, Nblocks)

% visualize pcs
fh = figure('name', 'Principal components, Gabor');
set(gcf, 'Position', [311   314   954   753], 'Color', [1 1 1]);
for i = 1:20
    pcselect = zeros(1,size(pc,2)+1);
    pcselect(i) = 1;
    subplot(4,5,i)
    im = showGaborDST_ROTATE(G, pcselect', pc, Nblocks);
    imagesc(im(1:4:end,1:4:end)); axis('square'); axis('off'); title(i)
    colormap(gray(256)); drawnow
    drawnow;
end
