'''
    A minimal data-version-control implementation.
    
    Traverses project folders, finds files by extension (e.g., *.pth.tar)
    and tracks them, adding <filename>.dvc with the following info:
        
        filename: path to file relative to project root
        md5: the md5sum
        size: filesize in bytes
        full_path: full path to file (locally)
        remote_path: path to remote store
        
     e.g. adds ./EXPERIMENTS_MODELS/weights/mcnn_pytorch_v6.pth.tar.dvc with contents:
     
         path: ./EXPERIMENTS_MODELS/weights/mcnn_pytorch_v6.pth.tar
         md5: 3ffb299876616db7ebdbb321e943db4c
         size: 795131237
         full_path: /home/jovyan/work/DropboxProjects/UNETS/EXPERIMENTS_MODELS/weights/mcnn_pytorch_v6.pth.tar
         remote_path: https:/www.dropbox.com/work/DevboxSync/George/Projects/UNETS/EXPERIMENTS_MODELS/weights/mcnn_pytorch_v6.pth.tar
    
     This .dvc file can then be committed to version control.
     
     python dvc.py track_files --overwrite True
     python dvc.py track_file ./EXPERIMENTS_MODELS/weights/alexnet_imagenette.pth.tar --overwrite True
     
'''
import os, argparse
import fire
import yaml
import hashlib
import pathlib
from omegaconf import OmegaConf
from glob import glob
from pathlib import Path
from pdb import set_trace
from collections import OrderedDict
from fastprogress.fastprogress import progress_bar

current_file = pathlib.Path(__file__).parent.resolve()
project_name = Path(current_file).name 
dropbox_root = Path(os.path.join('https://www.dropbox.com/work/DevboxSync/George/Projects', project_name))

parser = argparse.ArgumentParser(description='Custom Data Version Control')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

def compute_md5sum(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
        
    return file_hash.hexdigest()

def track_file(filename, overwrite=False):
    
    outfile = filename + '.dvc'
    if os.path.exists(outfile) and overwrite==False:
        print(f'file exists (skipping): {outfile}')
        return
    else:
        print(f'tracking: {outfile}')
        
    local_path = Path(filename).resolve()
    remote_path = dropbox_root/Path(filename)
    md5 = compute_md5sum(filename)
    size = os.path.getsize(filename)
    
    dict_file = OrderedDict([
        ('filename', str(Path(filename).name)),
        ('md5', md5),
        ('size', size),
        ('path', filename),
        ('full_path', str(local_path)),
        ('remote_path', str(remote_path))
    ])
    
    with open(outfile, 'w') as fp:
        #OmegaConf.save(config=dict(dict_file), f=fp.name)
        for k,v in dict_file.items():
            fp.write(f'{k}: {v}\n')

def track_files(root_dir='.', pattern='**/*.pth.tar', overwrite=False):
    files = list(glob(os.path.join(root_dir, pattern), recursive=True))
    for filename in progress_bar(files):
        track_file(filename, overwrite=overwrite)

if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)