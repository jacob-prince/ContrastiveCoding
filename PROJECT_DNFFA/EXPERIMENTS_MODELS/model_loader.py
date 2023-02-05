from PROJECT_DNFFA.EXPERIMENTS_MODELS.models import barlow_twins

def load_model(arch, task, pretrained=True):
    
    if arch == 'alexnet' and task == 'barlow-twins':
        if pretrained is True:
            model, state_dict = barlow_twins.alexnet_gn_barlow_twins(pretrained=True)
        else:
            model, state_dict = barlow_twins.alexnet_gn_barlow_twins(pretrained=False)
            
    model.arch = arch
    model.task = task
    model.pretrained = pretrained
    
    model.model_str = f'{arch}-{task}'
    if model.pretrained is False:
        model.model_str = f'{model.model_str}-random'
    
    return model, state_dict