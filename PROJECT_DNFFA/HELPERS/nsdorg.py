import numpy as np
import os
from os.path import join, exists
import nibabel as nib
import pandas as pd
from IPython.core.debugger import set_trace
from PROJECT_DNFFA.HELPERS import paths
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
