from os.path import exists
import torch
import torchvision
from torchvision import transforms
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

        transform = ImageClassification(crop_size=224)
        
        # todo: find where this is
        state_dict = None
        
    elif 'alexnet-barlow-twins' in model_name:
        
        if 'random' in model_name:
            model, state_dict = barlow_twins.alexnet_gn_barlow_twins(pretrained=False)
        else:
            model, state_dict = barlow_twins.alexnet_gn_barlow_twins(pretrained=True)
        
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
            ])
        
    return model, transform, state_dict

# todo: refactor/remove in favor of more general feature extraction fn
def get_NSD_alexnet_activations(image_data):
    
    model, transform, _ = load_model('alexnet-supervised')

    activations = dict()

    for partition in ['train','test']:

        activations[partition] = dict()

        # transform the images
        X = transform(torch.from_numpy(image_data[partition].transpose(0,3,1,2)))

        model_history = tl.get_model_activations(model, X, which_layers='all')

        for layer in progress_bar(model_history.layer_labels):
            activations[partition][layer] = model_history[layer].tensor_contents.detach().numpy()
            
    return activations

def get_layer_group(model_name, layer_list = []):
    
    if 'alexnet-supervised' in model_name:
        
        all_layer_names = ['conv2d_1_2', 'relu_1_3', 'maxpool2d_1_4', 
                           'conv2d_2_5', 'relu_2_6', 'maxpool2d_2_7', 
                           'conv2d_3_8', 'relu_3_9', 
                           'conv2d_4_10', 'relu_4_11', 
                           'conv2d_5_12', 'relu_5_13', 'maxpool2d_3_14', 
                           'linear_1_19', 'relu_6_20', 
                           'linear_2_23', 'relu_7_24', 
                           'linear_3_25']
        
    elif 'alexnet-barlow-twins' in model_name:
        all_layer_names = ['conv2d_1_8',
                            'groupnorm_1_9',
                             'relu_1_10',
                             'maxpool2d_1_11',
                             'conv2d_2_12',
                             'groupnorm_2_13',
                             'relu_2_14',
                             'maxpool2d_2_15',
                             'conv2d_3_16',
                             'groupnorm_3_17',
                             'relu_3_18',
                             'conv2d_4_19',
                             'groupnorm_4_20',
                             'relu_4_21',
                             'conv2d_5_22',
                             'groupnorm_5_23',
                             'relu_5_24',
                             'maxpool2d_3_25',
                             'linear_1_28',
                             'batchnorm_1_29',
                             'relu_6_30',
                             'linear_2_31',
                             'batchnorm_2_32',
                             'relu_7_33',
                             'linear_3_34',
                             'batchnorm_3_35',
                             'output_1_36']

    if layer_list == []:
        layers_to_analyze = all_layer_names
    else:
        for lay in layer_list:
            assert(lay in all_layer_names)
        layers_to_analyze = layer_list
        
    return layers_to_analyze