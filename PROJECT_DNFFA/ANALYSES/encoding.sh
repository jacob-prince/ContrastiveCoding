#!/bin/bash

for s in {1..8}
do
    for roi in EBA
    do
        python3 encoding.py --model-name=alexnet-barlow-twins --units-for-encoding=vpnl-floc-bodies --ROI=$roi --subj=subj0$s
    done
done
