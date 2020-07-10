from __future__ import absolute_import

import os
from got10k.experiments import *

import sys
sys.path.append('.')
from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50(origin).pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('~/dataset/otb100')
    e = ExperimentOTB(root_dir, version=2015)
    e.run(tracker)
    e.report([tracker.name])
