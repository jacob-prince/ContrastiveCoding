import argparse
from os.path import exists
import torch
import torchvision
from torchvision import datasets
import torchlens as tl
import numpy as np
import copy
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from fastprogress import progress_bar
from IPython.core.debugger import set_trace

from PROJECT_DNFFA.HELPERS import paths, statsmodels, nnutils, plotting

parser = argparse.ArgumentParser(description='Identify DNN domain-selective units')

parser.add_argument('--model-name', default='alexnet-supervised', 
                    type=str, help='model whose features will be analyzed')

parser.add_argument('--selective-units', default='vpnl-floc-faces', 
                    type=str, help='localizer imageset to use + domain to extract')

parser.add_argument('--FDR-p', default=0.05, 
                    type=float, help='FDR correction alpha value')

parser.add_argument('--overwrite', default=False, 
                    type=bool, help='overwrite selectivity dicts?')

parser.add_argument('--verbose', default=True, 
                    type=bool, help='show print statements and plots?')


def main():
    
    args = parser.parse_args()
    
    selective_unit_dict = get_model_selective_units(args.model_name, 
                                                    args.selective_units,
                                                    args.FDR_p,
                                                    args.overwrite,
                                                    args.verbose)
    
    return


def get_model_selective_units(model_name, selective_units_to_extract, 
                              FDR_p = 0.05, overwrite = False, verbose = True):
    
    savefn = f"{paths.selective_unit_dir()}/{model_name}_{selective_units_to_extract}_FDR-{str(FDR_p)[2:]}.npy"
    
    target_layers = nnutils.get_layer_group(model_name)

    floc_imageset = '-'.join(selective_units_to_extract.split('-')[:2])
    target_domain = selective_units_to_extract.split('-')[-1]

    print(savefn, '\n', target_domain)
    
    if exists(savefn) and overwrite is False:
    
        selective_units = np.load(savefn,allow_pickle=True).item()
    
    else:
    
        model, transforms, _ = nnutils.load_model(model_name)
        
        # ensure that all relus are converted to inplace=False
        nnutils.convert_relu(model)

        floc_imageset_dir = f'{paths.imageset_dir()}/{floc_imageset}'

        floc_dataset = datasets.ImageFolder(root = floc_imageset_dir, transform = transforms)

        # print some info -> verify correct # imgs, etc
        categ_idx = np.array(floc_dataset.targets)

        if floc_imageset == 'vpnl-floc':

            floc_categs = np.array(['adult','body','car','child','corridor','house','instrument','limb','number','scrambled','word'])
            floc_domains = np.array(['faces','bodies','objects','scenes','characters','scrambled'])
            floc_categ_domain_ref = np.array([0,1,2,0,3,3,2,1,4,5,4])
            categ_nimg = 144

            all_domain_idx = np.repeat(floc_categ_domain_ref,categ_nimg)

        elif floc_imageset == 'classic-categ':

            floc_domains = np.array([f"{domain.split('-')[1]}" for domain in floc_dataset.classes])
            floc_categ_domain_ref = np.array([0,1,2,3,4,5,6])
            categ_nimg = 80

            all_domain_idx = np.repeat(floc_categ_domain_ref,categ_nimg)

        target_domain_val = np.squeeze(np.argwhere(floc_domains == target_domain))

        ## visualize, for sanity
        if verbose:

            print(floc_domains)
            print(floc_domains[target_domain_val], target_domain_val)
            print('# localizer images in target domain:')
            print(np.sum(all_domain_idx == target_domain_val))

            plt.figure()
            plt.plot(all_domain_idx)
            plt.yticks(np.arange(len(floc_domains)),floc_domains);
            plt.xlabel('floc img idx')
            plt.title('domain of each floc image');

        # data loader object is required for passing images through the network - choose batch size and num workers here
        data_loader = torch.utils.data.DataLoader(
            dataset=floc_dataset,
            batch_size=len(floc_dataset),
            num_workers=32,
            shuffle=False,
            pin_memory=False
        )

        image_tensors, _ = next(iter(data_loader))

        model_history = tl.get_model_activations(model, image_tensors, which_layers='all')
    
        ################

        selective_units = dict()

        for layer in progress_bar(target_layers):

            Y = model_history[layer].tensor_contents.detach().numpy()

            if Y.ndim > 2:
                Y = Y.reshape(Y.shape[0],Y.shape[1]*Y.shape[2]*Y.shape[3])

            print(Y.shape)

            selective_units[layer] = compute_selectivity(Y, 
                                                         all_domain_idx, 
                                                         target_domain_val,
                                                         FDR_p,
                                                         verbose)

            np.save(savefn, selective_units, allow_pickle=True)
            
    if verbose:
        plotting.plot_selective_unit_props(selective_units, model_name, target_domain, floc_imageset, FDR_p)
    
    return selective_units

def compute_selectivity(Y, all_domain_idx, target_domain_val, FDR_p, verbose=True):
    
    assert(Y.ndim == 2)
    n_neurons_in_layer = Y.shape[1]
    
    unique_domain_vals = np.unique(all_domain_idx)
    
    target_domain_idx = all_domain_idx == target_domain_val
    
    # get data from curr domain
    Y_curr = copy.deepcopy(Y[target_domain_idx])
    
    if verbose:
        print(f'\t{np.sum(target_domain_idx)} of {len(all_domain_idx)} images are from the target domain ({target_domain_val})')
        print(f'\tsize of layer is {n_neurons_in_layer} units.')
        print(f'\tshape of Y_curr is {Y_curr.shape}')
        
    dom_pref_rankings = []
    dom_tvals_unranked = []
    
    first_comparison = True
    
    for this_domain_val in unique_domain_vals:
        
        # skip if test was domain vs. same domain
        if this_domain_val != target_domain_val:
        
            Y_test = copy.deepcopy(Y[all_domain_idx==this_domain_val])
            
            # calculate t and p maps
            t,p = stats.ttest_ind(Y_curr, Y_test, axis=0)
            
            # determine which neurons remain significant after FDR correction
            # https://stats.stackexchange.com/questions/63441/what-are-the-practical-differences-between-the-benjamini-hochberg-1995-and-t
            reject, pvals_corrected, _, _ = statsmodels.multipletests(p, alpha=FDR_p, method='FDR_by', 
                                is_sorted=False, returnsorted=False)
            
            # sort indices according to the t map
            # sort the neuron indices according to the t map
            dom_pref_ranking = np.flip(np.argsort(t))
            
            # accumulate tvals
            dom_tvals_unranked.append(t)

            # assert that no indices are repeated
            assert(len(np.unique(dom_pref_ranking)) == len(dom_pref_ranking))
            
            # calculate the size of the significant ROI
            dom_nsig = np.sum(np.logical_and(reject == True, t > 0))
            
            # store neuron selectivity rankings
            dom_pref_rankings.append(dom_pref_ranking)
            
            # if the first comparison... 
            if first_comparison:
                dom_neurons = dom_pref_ranking[:dom_nsig] # create dnn selective region
                if verbose:
                    print(f'\tsize of initial region is {len(dom_neurons)} units. comparison is {target_domain_val} vs. {this_domain_val}')
                first_comparison = False
            else: # slim down the region
                # using pandas intersection function because np.intersect1d returns sorted
                dom_neurons = pd.Index.intersection(pd.Index(dom_neurons), pd.Index(dom_pref_ranking[:dom_nsig]), sort = False).to_numpy()
                if verbose:
                    print(f'\tnew size of region is {len(dom_neurons)} units. comparison is {target_domain_val} vs. {this_domain_val}')
                    
    # calculate the size of the significant ROI
    dom_nsig = len(dom_neurons)
    
    if verbose:
        print(f'\t\tfinal size of region for {target_domain_val} is {dom_nsig} units.')
        
    # figure out which neurons are most selective across all categ pair comparisons...
    dom_ranking_score = np.zeros(n_neurons_in_layer)

    # "score" each index using the sort function
    for j in range(len(dom_pref_rankings)):
        dom_score = np.argsort(dom_pref_rankings[j])

        # accumulate scores
        dom_ranking_score = dom_ranking_score + dom_score

    # get the final selectivity indices - lowest score = most selective
    dom_ranking_score_final = np.argsort(dom_ranking_score)
    
    # get the average t-val for all units across all comparisons
    mean_tvals_unranked = np.nanmean(np.vstack(dom_tvals_unranked), axis = 0)
    
    # log
    out = dict()
    out['n_selective'] = dom_nsig
    out['prop_selective'] = dom_nsig / n_neurons_in_layer
    out['selective_idx'] = dom_neurons
    out['selective_rankings'] = dom_ranking_score_final
    out['mean_tvals_unranked'] = mean_tvals_unranked

    
    # create binary mask for lesioning, where all indices except selective units are 1
    lesioning_mask = np.ones(Y.shape[1],)
    lesioning_mask[dom_neurons] = 0
    
    out['lesioning_mask'] = lesioning_mask
   
    return out

if __name__ == '__main__':
    main()
     