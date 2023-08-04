
clear; clc; close all;

addpath('Code-GistModel')

opts = struct('FeatureModel',{{'Gabor', 'GistPC'}}, ...
              'Pixel_SmallSize', 64, ...
              'Gabor_NBlocks', 16, ...
              'Gabor_NrOrientationsPerScale', [12 8 6 4], ...
              'GistPC_NumPC', 50, ...
              'DistanceFunction', {{'euclidean', 'correlation', 'cosine'}}, ...
              'MaxNumClasses', 50, ...
              'ClassColors', '', ...
              'ClassifyAlgorithm', {{'NaiveBayes', 'SVM'}});

imdir = '/Users/jacobprince/KonkLab Dropbox/Jacob Prince/DevBoxSync/Projects/DNFFA/PROJECT_DNFFA/NOTEBOOKS/analysis_outputs/3d-LowLevel/';

imfns = {'special515_images.mat';
        'subj01_train-nonshared1000-3rep-batch0_images.mat';
        'subj02_train-nonshared1000-3rep-batch0_images.mat';
        'subj03_train-nonshared1000-3rep-batch0_images.mat';
        'subj04_train-nonshared1000-3rep-batch0_images.mat';
        'subj05_train-nonshared1000-3rep-batch0_images.mat';
        'subj06_train-nonshared1000-3rep-batch0_images.mat';
        'subj07_train-nonshared1000-3rep-batch0_images.mat';
        'subj08_train-nonshared1000-3rep-batch0_images.mat'};

s = 1;
for i = 1:length(imfns)
    if contains(imfns{i}, 'subj')

        gist_savefn = [imdir 'subj0' num2str(s) '_GistPC.mat'];
        gabor_savefn = [imdir 'subj0' num2str(s) '_Gabor.mat'];
        s = s+1;
    else
        gist_savefn = [imdir 'special515_GistPC.mat'];
        gabor_savefn = [imdir 'special515_Gabor.mat'];

    end
        disp(gist_savefn)
        disp(gabor_savefn)


   [Gabor, GistPC, fh] = computeGaborAndGistFeatures(imfns{i},...
                                                opts.Gabor_NBlocks, ...
                                                opts.Gabor_NrOrientationsPerScale, ...
                                                opts.GistPC_NumPC);

   save(gist_savefn, 'GistPC')
   save(gabor_savefn, 'Gabor')

   tag = split(imfns{i},'_');
   tag = tag{1};
   print(fh, '-dpng', [tag '_summary.png']);

end



%%

%imdirs = struct2table(dir('/Users/jacobprince/Dropbox (KonkLab)/Research-Prince/Deepnet-FFA/ImageSets/'));
%imdirs = imdirs(4:end,:);

% outputs = struct();
% 
% for i = 1:size(imdirs,1)
% 
%     imdir = imdirs(i,:).name;
%     imdir = strjoin([imdirs(i,:).folder '/' imdir],'');
%     imdir_label = strsplit(imdir,'/');
%     imdir_label = imdir_label{end};
%     disp(imdir_label)
% 
%     imtable = struct2table(dir(imdir));
%     imtable = imtable(imtable.isdir==0,:);
% 
%     impaths = cellfun(@(x,y) [x '/' y], imtable.folder, imtable.name,'un',0);
% 
%     imnums = [];
% 
%     for j = 1:length(impaths)
%         tmp = strsplit(impaths{j},'/');
%         tmp = str2num(tmp{end}(1:end-4));
%         imnums = [imnums; tmp];
%     end
%     
%     [~,order] = sort(imnums);
% 
%     impaths = impaths(order);
% 
%     [Gabor, GistPC, fh] = computeGaborAndGistFeatures(impaths,...
%                                                 opts.Gabor_NBlocks, ...
%                                                 opts.Gabor_NrOrientationsPerScale, ...
%                                                 opts.GistPC_NumPC);
% 
%     outputs.(['gabor_features_' imdir_label]) = Gabor.featureMatrix;
%     outputs.(['gistPC_features_' imdir_label]) = GistPC.featureMatrix;
% 
% end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%
% 
% %%
% 
% 
% % imdir = '/Users/jacobprince/Dropbox (KonkLab)/Research-Prince/Deepnet-FFA/ImageSets/special515';
% % 
% % imstruct = struct2table(dir(imdir));
% % 
% % imstruct = imstruct(imstruct.isdir==0,:);
% % 
% % impaths = cellfun(@(x,y) [x '/' y], imstruct.folder, imstruct.name,'un',0);
% % 
% % 
% % 
% % 
% % %%
% % 
% % outputs = struct();
% % 
% % %%
% % 
% % [Gabor, GistPC, fh] = computeGaborAndGistFeatures(impaths,...
% %                                                 opts.Gabor_NBlocks, ...
% %                                                 opts.Gabor_NrOrientationsPerScale, ...
% %                                                 opts.GistPC_NumPC);
% % 
% % %% 
% % 
% % ordering = cellstr(load('ordering.mat').ordering);
% % currorder = imstruct.name;
% % 
% % for i = 1:length(ordering)
% %     ordering{i} = ordering{i}(1:end-4);
% %     currorder{i} = currorder{i}(1:end-4);
% % end
% % 
% % ordering = cellfun(@str2num,ordering);
% % currorder = cellfun(@str2num,currorder);
% % 
% % %%
% % 
% % reorderidx = [];
% % 
% % for i = 1:length(ordering)
% %     reorderidx  = [reorderidx; find(currorder == ordering(i))];
% % end
% % %%
% % 
% % gabor_features = Gabor.featureMatrix(reorderidx,:);
% % gist_features = GistPC.featureMatrix(reorderidx,:);
% % 
% % figure;
% % imagesc(gabor_features); colorbar
% % figure;
% % imagesc(gist_features); colorbar
% % 
% % gabor_rdm = pdist(gabor_features,'correlation')';
% % gist_rdm = pdist(gist_features,'correlation')';
% % 
% % figure;
% % imagesc(squareform(gabor_rdm)); colorbar
% % title('RDM over Gabor features (515 test stimuli)')
% % figure;
% % imagesc(squareform(gist_rdm)); colorbar
% % title('RDM over GistPC features (515 test stimuli)')
% % 
