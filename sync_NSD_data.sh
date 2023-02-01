#!/bin/bash

DATADIR=~/work/DataLocal-w/NSD 

# syncing betas
aws s3 sync s3://natural-scenes-dataset/nsddata_betas $DATADIR/nsddata_betas \
            --exclude "*func1mm*" \
            --exclude "*MNI*" \
            --exclude "*betas_assumehrf*" \
            --exclude "*betas_fithrf/*" \
            --exclude "*betas_fithrf_GLMdenoise/*" \
            --exclude "*betas*session*nii.gz"

# syncing anatomical data and other important files in 'nsddata'
aws s3 sync s3://natural-scenes-dataset/nsddata $DATADIR/nsddata

# syncing the stimuli
aws s3 sync s3://natural-scenes-dataset/nsddata_stimuli $DATADIR/nsddata_stimuli


