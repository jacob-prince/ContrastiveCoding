import cortex
import numpy as np
import matplotlib.pyplot as plt
from PROJECT_DNFFA.HELPERS import nsdorg
from IPython.core.debugger import set_trace

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
                     with_colorbar=colorbar,
                     recache=False)
    plt.title(roi_group,fontsize=44)
    plt.show()
    
    return plot_data

def plot_selective_unit_props(selective_units, model_name, target_domain, floc_imageset, FDR_p):
    
    target_layers = list(selective_units.keys())
    
    if floc_imageset == 'vpnl-floc':

        colors = [('faces','tomato'),
                  ( 'bodies','orange'),
                  ('objects','dodgerblue'),
                  ('scenes','limegreen'),
                  ( 'characters','purple'),
                  ( 'scrambled','navy')]

    elif floc_imageset == 'classic-categ':

        colors = [('Faces','tomato'),
                  ('Bodies','orange'),
                  ('Scenes','limegreen'),
                  ('Words','purple'),
                  ('Objects','dodgerblue'),
                  ('Scrambled','dimgray')]

    floc_colors = [x[1] for x in colors]
    floc_domains = [x[0] for x in colors]
    
    idx = np.squeeze(np.argwhere(np.array(floc_domains) == target_domain))

    plt.figure(figsize=(8,4))

    domain_props = []
    for i in range(len(target_layers)):
        domain_props.append(selective_units[target_layers[i]]['prop_selective'])

    domain_props = np.vstack(domain_props)

    print(domain_props.shape)

    for n in range(domain_props.shape[1]):
        plt.plot(domain_props[:,n],label=target_domain,color=colors[idx][1]);

    plt.xticks(np.arange(len(target_layers)),np.array(target_layers),rotation=90);
    plt.title(f'proportion of domain-selective units by layer (FDR_p = {FDR_p})\nmodel: {model_name}\nfloc set: {floc_imageset}')
    plt.grid('on')
    # get rid of the frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.ylim([0,0.6])
    plt.legend()
    plt.show()
    
    return
