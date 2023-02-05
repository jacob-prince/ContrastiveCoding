from PROJECT_DNFFA.HELPERS import paths
from PROJECT_DNFFA.ANALYSES import train
import torch
from torch import nn
import copy
import wandb

N_IMAGENET_CLASSES = 1000

def train_readout_layer(model, readout_from):
    
    # deal with args
    args = train.arg_helper()
    args.readout_from = readout_from
    args.wandb_repo = 'DNFFA'
    
    
    # determine which readout layer is being used and modify model accordingly
    readout_model = append_readout_layer(model, readout_from)
    
    # prepare ffcv dataloaders
    train_loader, val_loader = train.get_ffcv_dataloaders(args)
    
    # main training loop
    readout_model = train.main_training_loop(readout_model, copy.deepcopy(train_loader), copy.deepcopy(val_loader), args)
    
    return readout_model

def append_readout_layer(model, readout_from): 
    
    # handle models on a case-by-case basis
    if model.arch == 'alexnet' and model.task == 'barlow-twins':
        
        if readout_from == 'fc6':
            in_features = 4096
            projector_idx = 0
        if readout_from == 'relu6':
            in_features = 4096
            projector_idx = 2
        if readout_from == 'fc7':
            in_features = 4096
            projector_idx = 3
        if readout_from == 'relu7':
            in_features = 4096
            projector_idx = 5
    
        # create a copy of the model to avoid modifying the original
        readout_model = copy.deepcopy(model)
        
        # linear readout
        readout_model.readout = nn.Linear(in_features=in_features, 
                                  out_features=N_IMAGENET_CLASSES, 
                                  bias=False)
        # reset the weight matrix
        readout_model.readout.weight.data.normal_(mean=0.0, std=0.01)
        
        # delete the batch norm module (to avoid confusion)
        delattr(readout_model, 'bn')

        # chop off everything up until the readout layer plus the next layer
        readout_model.projector = readout_model.projector[:projector_idx + 1]

        # add new forward function to the BarlowTwins instance as a class method
        bound_method = alexnet_readout_forward_projector.__get__(readout_model, 
                                                                 readout_model.__class__)
    
        # set the new forward function
        setattr(readout_model, 'forward', bound_method)
        
        #print(readout_model)
        
    # only the readout layer needs gradients
    for p, param in enumerate(readout_model.parameters()):
        if p == len(list(readout_model.parameters())) - 1:
            param.requires_grad = True
        else:
            param.requires_grad = False   
            
    return readout_model

def alexnet_readout_forward_projector(self, x):
    z = self.readout(self.projector(self.flatten(self.backbone(x))))
    return z
    
    