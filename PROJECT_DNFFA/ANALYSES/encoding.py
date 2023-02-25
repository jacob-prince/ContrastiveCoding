from PROJECT_DNFFA.HELPERS import paths, nsdorg, plotting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from fastprogress import progress_bar
import scipy.stats as stats
from IPython.core.debugger import set_trace
from multiprocessing import Process

import torch
import torchvision
import torchlens as tl

from torchvision.transforms._presets import ImageClassification

nsddir = paths.nsd()

xfm = 'func1pt8_to_anat0pt8_autoFSbbr'

def get_voxel_group(subj, space, voxel_group, ncsnr_threshold, roi_dfs, draw_plot = True):
    
        # need to define two main things: which rows of the roi_df(s) are we encoding, and which images are we using as train/test set

    include_idx = dict()

    if space == 'func1pt8mm':
        raise ValueError('func1pt8mm not implemented yet.')

    elif space == 'nativesurface':
        
        hemis = ['lh', 'rh']

        # liberal mask of visual cortex
        if voxel_group == 'nsdgeneral':

            for h, hemi in enumerate(hemis):
                include_idx[hemi] = np.logical_and(roi_dfs[h][f'{hemi}.nsdgeneral'].values == 1,
                                                   roi_dfs[h][f'{hemi}.ncsnr'].values > ncsnr_threshold)

        elif voxel_group == 'FFA-1':

            for h, hemi in enumerate(hemis):
                include_idx[hemi] = np.logical_and(np.isin(roi_dfs[h][f'{hemi}.floc-faces.label'].values, 'FFA-1'),
                                                   roi_dfs[h][f'{hemi}.ncsnr'].values > ncsnr_threshold)


        include_idx['full'] = np.concatenate((include_idx['lh'], include_idx['rh']))
        
    ### plot 
    
    plot_data = include_idx['full']

    if draw_plot:
        volume = plotting.plot_ROI_flatmap(subj,space,
                                            f'# total voxels for {subj}, {voxel_group}: {np.sum(plot_data)}'
                                           ,plot_data,vmin=np.min(plot_data),
                                                      vmax=np.max(plot_data))
        
    return include_idx
    

def load_encoding_betas(subj, space, voxel_group, ncsnr_threshold = 0.2,
                        beta_version = 'betas_fithrf_GLMdenoise_RR'):
    
    
    betadir = f'{nsddir}/nsddata_betas/ppdata/{subj}/{space}/{beta_version}'        
        
    stim_info_fn = f'{nsddir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    stim_info_df = pd.read_csv(stim_info_fn)
    
    roi_dfs = nsdorg.load_voxel_info(subj, space, beta_version)
    
    if space == 'nativesurface':
        
        hemis = ['lh', 'rh']

        for h, hemi in enumerate(hemis):
            indices = np.array([f'({hemi}, {i})' for i in np.arange(roi_dfs[h].shape[0])])
            roi_dfs[h].index = indices
    elif space == 'func1pt8mm':
        raise ValueError('func1pt8mm not implemented yet.')
                    
    ###############
        
    include_idx = get_voxel_group(subj, space, voxel_group, ncsnr_threshold, roi_dfs)
    
    nv = dict()
    nincl = dict()

    if space == 'nativesurface':

        for h, hemi in enumerate(hemis):
            nv[hemi] = roi_dfs[h].shape[0]
            nincl[hemi] = len(np.squeeze(np.argwhere(include_idx[hemi])))
                              
    elif space == 'func1pt8mm':
        raise ValueError('func1pt8mm not implemented yet.')
        
        
    ####### load betas
    
    betafiles = os.listdir(betadir)
    betafiles = np.sort([fn for fn in betafiles if 'betas_session' in fn])

    subj_nses = int(betafiles[-1][-7:-5])
    subj_nstim = 750 * subj_nses
    
    
    subj_betas = dict()

    for hemi in list(nv.keys()):

        subj_betas[hemi] = np.empty((subj_nstim, nincl[hemi]), dtype=float)
        print(subj_betas[hemi].shape)

    for hemi in hemis:

        start_idx = 0

        # get indices of included voxels
        load_idx = np.squeeze(np.argwhere(include_idx[hemi]))

        # get betafiles for this hemisphere
        if space == 'nativesurface':
            hemi_betafiles = np.sort([fn for fn in betafiles if hemi in fn])
        elif space == 'func1pt8mm':
            raise ValueError('func1pt8mm not implemented yet')

        # iterate through sessions
        for betafile in progress_bar(hemi_betafiles):

            #print(f'{betafile}, filling indices {start_idx} to {start_idx + 750}')

            f = h5py.File(f'{betadir}/{betafile}', 'r')

            # iterate through included voxels (for speed)
            for vox in range(nincl[hemi]):

                # add voxels to preallocated data matrices
                if space == 'nativesurface':
                    subj_betas[hemi][start_idx : start_idx + 750, vox] = stats.zscore(f['betas'][:, load_idx[vox]].astype(float) / 300)
                elif space == 'func1pt8mm':
                    raise ValueError('func1pt8mm not implemented yet')
                    
            start_idx += 750
            
    ######## group repetitions together
    
    subj_df = stim_info_df.iloc[stim_info_df[f'subject{subj[-1]}'].values==1]

    rep_indices = np.empty((subj_df.shape[0], 3), dtype=int)
    rep_cocos = []

    for i in range(rep_indices.shape[0]):

        # subtract 1 to get to 0 indexed
        rep_indices[i] = np.array([subj_df[f'subject{subj[-1]}_rep{r}'].values[i] for r in range(3)]) - 1

        rep_cocos.append(subj_df['cocoId'].values[i])

    rep_cocos = np.array(rep_cocos)
    
    # reshape brain data to group repetitions together
    for hemi in hemis:
    
        # conditions x repetitions x voxels/vertices
        subj_betas[hemi] = subj_betas[hemi][rep_indices]
        print(subj_betas[hemi].shape)
  
    return subj_betas, roi_dfs, include_idx, rep_cocos

def get_train_test_data(subj, space, subj_betas, rep_cocos, train_imageset, test_imageset):
    
    if space == 'nativesurface':
        hemis = ['lh','rh']
    elif space == 'func1pt8mm':
        hemis = ['full']
        
    stim_info_fn = f'{nsddir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    stim_info_df = pd.read_csv(stim_info_fn)
    
    subjs = [f'subj0{s}' for s in range(1,9)]
    annotations = nsdorg.load_NSD_coco_annotations(subjs, savedir = paths.nsd_coco_annots())
    
    coco_dict = nsdorg.get_coco_dict(subjs, annotations)

    encoding_cocos = dict()
    try:
        encoding_cocos['train'] = coco_dict[subj][train_imageset]
    except:
        encoding_cocos['train'] = coco_dict[train_imageset]
    try:
        encoding_cocos['test'] = coco_dict[subj][test_imageset]
    except:
        encoding_cocos['test'] = coco_dict[test_imageset]

    nc = dict()
    nc['train'] = len(encoding_cocos['train'])
    nc['test'] = len(encoding_cocos['test'])
    
    # access nsd stimuli
    stim_f = h5py.File(f'{paths.nsd_stimuli()}/nsd_stimuli.hdf5', 'r')
    dim = stim_f['imgBrick'].shape

    image_data = dict()
    brain_data = dict()

    for partition in ['train','test']:

        image_data[partition] = np.empty((nc[partition], dim[1], dim[2], dim[3]), dtype=np.uint8)
        brain_data[partition] = dict()

        for hemi in hemis:
            brain_data[partition][hemi] = np.empty((nc[partition], 3, subj_betas[hemi].shape[2]), dtype=float)

        for c, coco in enumerate(progress_bar(encoding_cocos[partition])):

            # where in the brain data does this coco live?
            idx10k = np.squeeze(np.argwhere(rep_cocos == coco))

            # where in the stimulus brick does this coco live?
            idx73k = stim_info_df.iloc[stim_info_df['cocoId'].values == coco]['nsdId'].values[0]

            image_data[partition][c] = stim_f['imgBrick'][idx73k]

            for hemi in hemis:
                brain_data[partition][hemi][c] = subj_betas[hemi][idx10k]

        for hemi in hemis:
            brain_data[partition][hemi] = np.mean(brain_data[partition][hemi], axis = 1)
            print(partition, hemi, image_data[partition].shape, brain_data[partition][hemi].shape)
            
    return image_data, brain_data
            

def get_alexnet_activations(image_data):
    
    # switch to eval mode
    model = torchvision.models.alexnet(weights='DEFAULT').eval()

    alexnet_transforms = ImageClassification(crop_size=224)

    activations = dict()

    for partition in ['train','test']:

        activations[partition] = dict()

        # transform the images
        X = alexnet_transforms(torch.from_numpy(image_data[partition].transpose(0,3,1,2)))

        model_history = tl.get_model_activations(model, X, which_layers='all')

        for layer in progress_bar(model_history.layer_labels):
            activations[partition][layer] = model_history[layer].tensor_contents.detach().numpy()
            
    return activations


def fit_save_encoding_model(savedir, batch_idx, voxel_idx):
    
    savefn = f'{savedir}/batch_{batch_idx}.npy'
    
    if not exists(savefn):
    
        mdl = Lasso(positive=True, alpha = 0.1, random_state = 365, selection = 'random', tol = 1e-3, fit_intercept=True)
        mdl.fit(X_train, y_train[:,voxel_idx])

        out = dict()
        out['voxel_ids'] = voxel_ids[voxel_idx]
        out['coef'] = mdl.sparse_coef_
        out['intercept'] = mdl.intercept_
        out['y_pred'] = mdl.predict(X_test)
        out['r'] = np.array([stats.pearsonr(y_test[:,xx], 
                                       out['y_pred'][:,x])[0] for x, xx in enumerate(voxel_idx.astype(int))])

        np.save(savefn, out, allow_pickle=True)
        
    else:
        print(f'file {savefn} already exists. done')



            
    

    
        
    