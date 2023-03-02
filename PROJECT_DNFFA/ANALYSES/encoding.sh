#!/bin/bash

for s in {1..8}
do
    for roi in VWFA-1 VWFA-2 OWFA
    do
        python3 encoding.py --model-name=alexnet-supervised --units-for-encoding=vpnl-floc-characters --ROI=$roi --subj=subj0$s
    done
done

for s in {1..8}
do
    for roi in OFA FFA-2 FFA-1
    do
        python3 encoding.py --model-name=alexnet-supervised --units-for-encoding=vpnl-floc-faces --ROI=$roi --subj=subj0$s
    done
done

for s in {1..8}
do
    for roi in FBA-1 FBA-2 EBA
    do
        python3 encoding.py --model-name=alexnet-supervised --units-for-encoding=vpnl-floc-bodies --ROI=$roi --subj=subj0$s
    done
done

for s in {1..8}
do
    for roi in OPA PPA
    do
        python3 encoding.py --model-name=alexnet-supervised --units-for-encoding=vpnl-floc-scenes --ROI=$roi --subj=subj0$s
    done
done
