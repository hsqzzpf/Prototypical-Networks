# Prototypical Networks for Few shot Learning in PyTorch
Simple alternative Implementation of Prototypical Networks for Few Shot Learning ([paper](https://arxiv.org/abs/1703.05175), [code](https://github.com/jakesnell/prototypical-networks)) in PyTorch.

## Performances (MiniImageNet Dataset)

Download images from: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

Put the upzipped `images` folder inside `miniImageNet` folder

We are trying to reproduce the reference paper performaces, we'll update here our best results. 

| Model | 1-shot (5-way Acc. 100 iters) | 5-shot (5-way Acc. 100 iters) | 1-shot (5-way Acc. 30 iters) | 5-shot (5-way Acc. 20 iters)|
| --- | --- | --- | --- | --- |
| Reference Paper | - | - | 49.42% | 68.20%|
| This repo | 42.48%* | 64.7%** | 38.18%° | 60.87%°°|

\* achieved by running `python train_miniImageNet.py --cuda -nsTr 1 -nsVa 1` (30 epochs)

\*\* achieved by running `python train_miniImageNet.py --cuda` (30 epochs)

° achieved by running `python3 train_miniImageNet.py --cuda -nsTr 1 -nsVa 1 -its 30` (50 epochs)

°° achieved by running `python train_miniImageNet.py --cuda -its 20` (50 epochs)



