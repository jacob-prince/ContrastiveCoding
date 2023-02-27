from os.path import exists
import torch
import torchvision
import torchlens as tl
import numpy as np
import copy
import pandas as pd
import scipy.stats as stats
from fastprogress import progress_bar

from torchvision.transforms._presets import ImageClassification

from PROJECT_DNFFA.EXPERIMENTS_MODELS.models import barlow_twins
from PROJECT_DNFFA.HELPERS import paths

def load_model(model_name):
    
    if model_name == 'alexnet-supervised':
        
        # switch to eval mode
        model = torchvision.models.alexnet(weights='DEFAULT').eval()

        transforms = ImageClassification(crop_size=224)
        
        # todo: find where this is
        state_dict = None
        
    elif model_name == 'alexnet-barlow-twins':
        
        model, state_dict = barlow_twins.alexnet_gn_barlow_twins(pretrained=True)
        
        # todo: find where this is
        transforms = None
        
    elif model_name == 'alexnet-barlow-twins-random':
        
        model, state_dict = barlow_twins.alexnet_gn_barlow_twins(pretrained=False)
        
        # todo: find where this is
        transforms = None
        
    return model, transforms, state_dict


def get_NSD_alexnet_activations(image_data):
    
    model, transforms, _ = load_model('alexnet-supervised')

    activations = dict()

    for partition in ['train','test']:

        activations[partition] = dict()

        # transform the images
        X = transforms(torch.from_numpy(image_data[partition].transpose(0,3,1,2)))

        model_history = tl.get_model_activations(model, X, which_layers='all')

        for layer in progress_bar(model_history.layer_labels):
            activations[partition][layer] = model_history[layer].tensor_contents.detach().numpy()
            
    return activations

def get_layer_group(model_name, layer_list = []):
    
    if model_name == 'alexnet-supervised':
        
        all_layer_names = ['conv2d_1_2', 'relu_1_3', 'maxpool2d_1_4', 
                           'conv2d_2_5', 'relu_2_6', 'maxpool2d_2_7', 
                           'conv2d_3_8', 'relu_3_9', 
                           'conv2d_4_10', 'relu_4_11', 
                           'conv2d_5_12', 'relu_5_13', 'maxpool2d_3_14', 
                           'linear_1_19', 'relu_6_20', 
                           'linear_2_23', 'relu_7_24', 
                           'linear_3_25']

        if layer_list == []:
            layers_to_analyze = all_layer_names
        else:
            for lay in layer_list:
                assert(lay in all_layer_names)
            layers_to_analyze = layer_list
        
    return layers_to_analyze