import cortex
import numpy as np
import matplotlib.pyplot as plt
from PROJECT_DNFFA import nsdorg

def plot_ROI_flatmap(subj, space, roi_group, included_voxels, mapper='nearest',vmin=0,vmax=1,cmap='Spectral_r',colorbar=True):

    if space == 'nativesurface':
        plot_data = cortex.Vertex(included_voxels, subj, cmap=cmap,
                                        vmin=vmax,
                                        vmax=vmin)
    elif space == 'func1pt8mm':
        subj_dims = nsdorg.get_subj_dims(subj)
        included_voxels = np.swapaxes(included_voxels.reshape(subj_dims),0,2)
        plot_data = cortex.Volume(included_voxels, subj, xfmname='func1pt8_to_anat0pt8_autoFSbbr', cmap=cmap,vmin=vmin,vmax=vmax)

    plt.figure()
    cortex.quickshow(plot_data,with_rois=False,with_labels=False,with_curvature=True,
                     curvature_contrast=0.3,
                     curvature_brightness=0.8,
                     curvature_threshold=True,
                     with_colorbar=colorbar)
    plt.title(roi_group,fontsize=44)
    plt.show()
    
    return plot_data
