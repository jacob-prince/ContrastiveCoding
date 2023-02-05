import numpy as np
import os
from os.path import join
import nibabel as nib
import pandas as pd
from PROJECT_DNFFA import paths

nsddir = paths.nsd()               
surfdir = f'{nsddir}/nsddata/freesurfer'
betadir = f'{nsddir}/nsddata_betas/ppdata'

def get_subj_dims(subj):
    fn = f'{nsddir}/nsddata/ppdata/{subj}/func1pt8mm/mean.nii.gz'
    return nib.load(fn).get_fdata().shape

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
