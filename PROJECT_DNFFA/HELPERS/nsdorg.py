import numpy as np
import os
from os.path import join, exists
import nibabel as nib
import pandas as pd
import scipy.stats as stats
import h5py
from IPython.core.debugger import set_trace
from PROJECT_DNFFA.HELPERS import paths, plotting
from fastprogress import progress_bar

from pycocotools.coco import COCO

nsddir = paths.nsd()      
annotdir = paths.full_coco_annots()
surfdir = f'{nsddir}/nsddata/freesurfer'
betadir = f'{nsddir}/nsddata_betas/ppdata'

def get_subj_dims(subj):
    fn = f'{nsddir}/nsddata/ppdata/{subj}/func1pt8mm/mean.nii.gz'
    return nib.load(fn).get_fdata().shape

def get_coco_class_name(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['supercategory'], cats[i]['name']
    return "None"

def get_coco_dict(subjs, annotations):
    
    coco_dict = dict()
    nsddir = paths.nsd()
    info_fn = f'{nsddir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    nsd_df = pd.read_csv(info_fn)

    shared1000_cocos = np.sort(nsd_df.iloc[nsd_df['shared1000'].values==True]['cocoId'].values)

    coco_dict['shared1000'] = shared1000_cocos

    coco_dict['special515'] = []

    for subj in subjs:

        coco_dict[subj] = dict()

        # get indices of subject-specific non-shared cocos
        coco_dict[subj]['shared'] = np.sort(np.intersect1d(np.unique(annotations[subj].index), coco_dict['shared1000']))
        coco_dict[subj]['nonshared'] = np.sort(np.setdiff1d(np.unique(annotations[subj].index), coco_dict['shared1000']))

        # get indices of cocos each subject viewed 3x
        a,b = np.unique(annotations[subj].index,return_counts=True)
        subj_3rep_cocos = np.unique(a[b==3])

        # get subset of 3x cocos that are shared/nonshared
        coco_dict[subj]['shared-3rep'] = np.sort(np.intersect1d(coco_dict['shared1000'], subj_3rep_cocos))
        coco_dict[subj]['nonshared-3rep'] = np.sort(np.setdiff1d(subj_3rep_cocos, coco_dict['shared1000']))

        #print(len(coco_dict[subj]['shared_3rep']),len(coco_dict[subj]['nonshared_3rep']))

        # get 1000 nonshared 3x cocos from each subject
        for batch in range(4):
            # all subjs have at least 4000 nonshared 3rep cocos
            coco_dict[subj][f'nonshared1000-3rep-batch{batch}'] = coco_dict[subj]['nonshared-3rep'][:4000][batch::4] 

        # get the special515 cocos
        if len(coco_dict['special515']) == 0:
            coco_dict['special515'] = subj_3rep_cocos
        else:
            coco_dict['special515'] = np.sort(np.intersect1d(coco_dict['special515'], subj_3rep_cocos))

    assert(len(coco_dict['special515']) == 515)
    assert(len(np.intersect1d(coco_dict['shared1000'], coco_dict['special515'])) == 515)

    #print(coco_dict[subj].keys())
    
    return coco_dict

def load_NSD_coco_annotations(subjs, savedir):
    
    annotations = dict()
    
    subsess = [40, 40, 32, 30, 40, 32, 40, 30]
    nstim = 30000
    nreps = 3
    
    info_fn = f'{nsddir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    df = pd.read_csv(info_fn)
    
    all_cocos = df['cocoId'].values
    
    first_subj = True
    
    for subj in subjs:
        
        if not exists(join(savedir,f'{subj}_annotation_df.pkl')):
        
            if first_subj:
                coco_train_annotations_path = f'{annotdir}/instances_train2017.json'
                coco_train = COCO(coco_train_annotations_path)

                coco_val_annotations_path = f'{annotdir}/instances_val2017.json'
                coco_val = COCO(coco_val_annotations_path)

                coco_train_captions_path = f'{annotdir}/captions_train2017.json'
                coco_train_captions = COCO(coco_train_captions_path)

                coco_val_captions_path = f'{annotdir}/captions_val2017.json'
                coco_val_captions = COCO(coco_val_captions_path)

                catIDs = coco_train.getCatIds()
                cats = coco_train.loadCats(catIDs)

                supercategs = list(np.unique([x['supercategory'] for x in cats]))

            s = int(subj[-1])

            this_subsess = subsess[s-1]
            #print(subj, 'nses', this_subsess)

            subj_ids = np.vstack((df[f'subject{subj[-1]}_rep0'].values, 
                                  df[f'subject{subj[-1]}_rep1'].values, 
                                  df[f'subject{subj[-1]}_rep2'].values)).T

            subj_cocos = []
            for i in progress_bar(range(1,750*this_subsess+1)):
                idx = np.argwhere(subj_ids == i)[0][0]
                subj_cocos.append(df.iloc[idx]['cocoId'])

            ##################

            subj_annots = dict()

            all_scats = []
            all_cats = []
            all_areas = []
            all_captions = []

            for cocoid in progress_bar(subj_cocos):

                this_scats = []
                this_cats = []
                this_areas = []

                annIds = coco_val.getAnnIds(imgIds=cocoid)
                anns = coco_val.loadAnns(annIds)

                if len(anns) == 0:
                    annIds = coco_train.getAnnIds(imgIds=cocoid)
                    anns = coco_train.loadAnns(annIds)

                for a in anns:
                    this_scats.append(get_coco_class_name(a['category_id'], cats)[0])
                    this_cats.append(get_coco_class_name(a['category_id'], cats)[1])
                    this_areas.append(a['area'])

                all_scats.append(this_scats)
                all_cats.append(this_cats)
                all_areas.append(this_areas)

                # get captions
                annIds_cap = coco_train_captions.getAnnIds(imgIds=cocoid)
                anns_cap = coco_train_captions.loadAnns(annIds_cap)

                if len(anns_cap) == 0:
                    annIds_cap = coco_val_captions.getAnnIds(imgIds=cocoid)
                    anns_cap = coco_val_captions.loadAnns(annIds_cap)

                captions = []
                for a in anns_cap:
                    captions.append(a['caption'])

                all_captions.append(captions)

            subj_annots['coco_supercategs'] = all_scats
            subj_annots['coco_categs'] = all_cats
            subj_annots['coco_areas'] = all_areas
            subj_annots['coco_captions'] = all_captions

            annotation_df = pd.DataFrame(subj_annots)
            annotation_df.index = subj_cocos

            print(subj, annotation_df.shape)

            annotation_df.to_pickle(join(savedir,f'{subj}_annotation_df.pkl'))
            
            annotations[subj] = annotation_df

            first_subj = False
            
        else:
            #print(subj,'already exists. loading...')
            
            annotations[subj] = pd.read_pickle(join(savedir,f'{subj}_annotation_df.pkl'))
        
    return annotations

def load_voxel_info(subj, space, beta_version):
    
    if space == 'func1pt8mm':
        roidir = f'{nsddir}/nsddata/ppdata/{subj}/{space}/roi'
        suffix = '.nii.gz'

    elif space == 'nativesurface':
        roidir = f'{surfdir}/{subj}/label'
        suffix = '.mgz'

    elif space == 'fsaverage':
        roidir = f'{surfdir}/fsaverage/label'
        suffix = '.mgz'

    if space == 'func1pt8mm':

        # get list of files that have roi info
        roifiles = np.sort([fn for fn in os.listdir(roidir) if not 'lh' in fn and not 'rh' in fn])

        # get list of files that are roi metadata
        annot_fns = []
        for fn in np.sort([fn for fn in os.listdir(f'{surfdir}/{subj}/label') if fn[-4:] == '.mgz' or '.ctab' in fn]):
            if '.mgz.ctab' in fn:
                annot_fns.append(fn)

        roi_dfs = []

        roidata = dict()

        roidata['ncsnr'] = nib.load(f'{betadir}/{subj}/{space}/{beta_version}/ncsnr.nii.gz').get_fdata().reshape(-1)
        
        for domain in ['faces','bodies','word','places']:
            tvals = nib.load(f'{nsddir}/nsddata/ppdata/{subj}/{space}/floc_{domain}tval.nii.gz').get_fdata().reshape(-1)
            if 'word' in domain:
                domain_ = 'words'
            else:
                domain_ = domain
            roidata[f'floc-{domain_}.tval'] = tvals

        for roifile in roifiles:

            X = nib.load(join(roidir,roifile)).get_fdata().reshape(-1)

            if np.ndim(X) == 1:
                roidata[roifile.split('.')[0]] = X.astype(float)

            if roifile.split('.')[0] + '.mgz.ctab' in annot_fns:

                # parse
                annots = pd.read_csv(join(f'{surfdir}/{subj}/label',
                                          roifile.split('.')[0] + '.mgz.ctab')).iloc[:,0].values
                annots_dict = dict()
                for an in annots:
                    vals = an.split(' ')
                    annots_dict[int(vals[0])] = vals[1].split('\t')[0]

                # add new col
                X_annots = []
                for x in X:

                    if int(x) <= 0:
                        X_annots.append('n/a')
                    else:
                        X_annots.append(annots_dict[int(x)])

                roidata[roifile.split('.')[0] + '.label'] = np.array(X_annots)

        roi_df = pd.DataFrame(data=roidata)

        roi_dfs.append(roi_df)

    elif space == 'nativesurface':

        # get list of files that have roi info
        roifiles = np.sort([fn for fn in os.listdir(roidir) if fn[-4:] == '.mgz' or '.ctab' in fn])

        # get list of files that are roi metadata
        annot_fns = []
        for fn in roifiles:
            if '.mgz.ctab' in fn:
                annot_fns.append(fn)

        hemis = ['lh','rh']

        roi_dfs = []

        for hemi in hemis:

            roidata = dict()

            roidata[f'{hemi}.ncsnr'] = np.squeeze(np.array(nib.load(f'{betadir}/{subj}/{space}/{beta_version}/{hemi}.ncsnr.mgh').get_fdata()).T)

            for roifile in roifiles:

                if hemi in roifile:
                    X = np.squeeze(np.array(nib.load(join(roidir,roifile)).get_fdata()).T)
                    #print(X.shape)

                    if np.ndim(X) == 1:
                        roidata['.'.join(roifile.split('.')[:2])] = X.astype(float)

                    if roifile[3:] + '.ctab' in annot_fns:

                        # parse
                        annots = pd.read_csv(join(roidir,roifile[3:] + '.ctab')).iloc[:,0].values
                        annots_dict = dict()
                        for an in annots:
                            vals = an.split(' ')
                            annots_dict[int(vals[0])] = vals[1].split('\t')[0]

                        # add new col
                        X_annots = []
                        for x in X:

                            if int(x) <= 0:
                                X_annots.append('n/a')
                            else:
                                X_annots.append(annots_dict[int(x)])

                        roidata['.'.join(roifile.split('.')[:2]) + '.label'] = np.array(X_annots)

            roi_df = pd.DataFrame(data=roidata)

            roi_dfs.append(roi_df)

    return roi_dfs

def get_voxel_group(subj, space, voxel_group, ncsnr_threshold, roi_dfs, plot = True):
    
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
                
        elif voxel_group == 'PPA':
            
            for h, hemi in enumerate(hemis):
                include_idx[hemi] = np.logical_and(np.isin(roi_dfs[h][f'{hemi}.floc-places.label'].values, 'PPA'),
                                                   roi_dfs[h][f'{hemi}.ncsnr'].values > ncsnr_threshold) 


        include_idx['full'] = np.concatenate((include_idx['lh'], include_idx['rh']))
        
    ### plot 
    
    plot_data = include_idx['full']

    if plot:
        volume = plotting.plot_ROI_flatmap(subj,space,
                                            f'# total voxels for {subj}, {voxel_group}: {np.sum(plot_data)}'
                                           ,plot_data,vmin=np.min(plot_data),
                                                      vmax=np.max(plot_data))
        
    return include_idx
    

def load_betas(subj, space, voxel_group, ncsnr_threshold = 0.2,
                        beta_version = 'betas_fithrf_GLMdenoise_RR',
                        plot = True):
    
    
    betadir = f'{nsddir}/nsddata_betas/ppdata/{subj}/{space}/{beta_version}'        
        
    stim_info_fn = f'{nsddir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    stim_info_df = pd.read_csv(stim_info_fn)
    
    roi_dfs = load_voxel_info(subj, space, beta_version)
    
    if space == 'nativesurface':
        
        hemis = ['lh', 'rh']

        for h, hemi in enumerate(hemis):
            indices = np.array([f'({hemi}, {i})' for i in np.arange(roi_dfs[h].shape[0])])
            roi_dfs[h].index = indices
    elif space == 'func1pt8mm':
        raise ValueError('func1pt8mm not implemented yet.')
                    
    ###############
        
    include_idx = get_voxel_group(subj, space, voxel_group, ncsnr_threshold, roi_dfs, plot = plot)
    
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
