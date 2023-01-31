# copy files from QNAP to local project folder
rsync -av --exclude .ipynb_checkpoints --exclude .DS_Store /home/jovyan/work/DataRemoteQNAP/Projects/UNETS-v1/models/ /home/jovyan/work/DropboxProjects/UNETS/PROJECT_UNETS/EXPERIMENTS_MODELS/models
