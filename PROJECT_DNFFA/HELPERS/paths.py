def base():
    return '/home/jovyan/work' 

def nsd():
    return f'{base()}/DataLocal-w/NSD'

def full_coco_annots():
    return f'{base()}/DataLocal-w/COCO/annotations'

def nsd_coco_annots():
    return f'{base()}/DropboxProjects/DNFFA/PROJECT_DNFFA/STIMULUS_SETS/fMRI/NSD_COCO_annotations'

def cifar10():
    return f'{base()}/DataLocal-w/ffcv-cifar10/'

def pycortex_db_NSD():
    return f'{base()}//DataLocal-w/pycortex_db_NSD'

def ffcv_imagenet1k_trainset():
    return f'{base()}/DataLocal-ro/imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop_includes_index.ffcv'

def ffcv_imagenet1k_valset():
    return f'{base()}/DataLocal-ro/imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop_includes_index.ffcv'

def training_checkpoint_dir():
    return f'{base()}/DropboxProjects/DNFFA/PROJECT_DNFFA/EXPERIMENTS_MODELS/checkpoints'
