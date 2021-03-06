# DivNoising

## Diversity Denoising with Fully Convolutional Variational Autoencoders Reimplementation for Deep Learning Course in Koç University

This repo will contain the Julia implementation of the paper Diversity Denoising with Fully Convolutional Variational Autoencoders with the Koç University Deep Learning Framework (Knet).

## Train
To start training, you can use DenoiSegMouse and W2S modules. First include them by calling '''include("train_mouse.jl")''' or ''include("train_W2S.jl")'''.
```
usage: <DenoiSegMouse> [--seed SEED] [--batchsize BATCHSIZE]
                 [--epochs EPOCHS] [--nf NF] [--lr LR]
                 [--gauss_std GAUSS_STD] [--kl_anneal KL_ANNEAL]
                 [--minvar MINVAR] [--dataset DATASET]

Fully Unsupervised  or Unsupervised Training on DenoiSeg Mouse and
DenoiSeg Mouse s&p datasets.

optional arguments:
  --seed SEED           random number seed: use a nonnegative int for
                        repeatable results (type: Int64, default: -1)
  --batchsize BATCHSIZE
                        minibatch size (type: Int64, default: 32)
  --epochs EPOCHS       number of epochs for training (type: Int64,
                        default: 1000)
  --nf NF               number of filters for first conv layer of
                        encoder (type: Int64, default: 32)
  --lr LR               initial learning rate (type: Float64, default:
                        0.001)
  --gauss_std GAUSS_STD
                        Gaussian noise assumption for the dataset,if
                        it is negative fully unsupervised traning is
                        performed (type: Float64, default: -1.0)
  --kl_anneal KL_ANNEAL
                        How many epochs to perform kl annealing
                        starting from first epoch. [To prevent
                        posterior collapse, KL annealing aproach might
                        be needed.] (type: Int64, default: 0)
  --minvar MINVAR       minimum allowed variance required for
                        [original implementation uses 9.0 for Mouse
                        and  1.0 for Mouse s&p] (type: Float64,
                        default: -1.0)
  --dataset DATASET     Choose 0 for  Mouse and 1 for Mouse s&p (type:
                        Int64, default: 0)
```

```
usage: <W2S> [--seed SEED] [--batchsize BATCHSIZE]
                 [--epochs EPOCHS] [--nf NF] [--lr LR]
                 [--channel CHANNEL] [--avg AVG] [--minvar MINVAR]

Fully Unsupervised Training on W2S datasets.

optional arguments:
  --seed SEED           random number seed: use a nonnegative int for
                        repeatable results (type: Int64, default: -1)
  --batchsize BATCHSIZE
                        minibatch size (type: Int64, default: 32)
  --epochs EPOCHS       number of epochs for training (type: Int64,
                        default: 1000)
  --nf NF               number of filters for first conv layer of
                        encoder (type: Int64, default: 32)
  --lr LR               initial learning rate (type: Float64, default:
                        0.001)
  --channel CHANNEL     W2S dataset channel to train 0,1 or 2 (type:
                        Int64)
  --avg AVG             W2S 1 or 16 corresponds to Avg1 and Avg 16
                        respectively (type: Int64)
  --minvar MINVAR       minimum allowed variance [original
                        implementation uses 1.0 for Avg16 9.0 for
                        Avg1] (type: Float64)

```
Please check train.ipynb for details. It shows how to reproduce my results.
## Results
| Dataset | Fully Unsupervised | Paper Reported Fully Unsupervised | Unsupervised| Paper Reported  Unsupervised |
|  ---         | ---       | ---        | ---  | --- |
|DenoiSeg Mouse| 34.11 dB  | 34.06 dB |34.19 dB| 34.13 dB | 
|DenoiSeg Mouse s&p | 36.20 dB    | 35.19 dB | __ | 36.21 dB| 
|DenoiSeg Flywing |24.79 dB | 24.92 dB     | 25.12 dB | 25.02 dB|
|W2S Ch.0 Avg 1 | 34.36 dB   | 34.24 dB   |  - | __ |
|W2S Ch.1 Avg 1 | 32.24 dB   | 32.22 dB   |  - | __ |
|W2S Ch.2 Avg 1 | 35.31 dB   | 35.24 dB   |  - | __ |
|W2S Ch.0 Avg 16| 39.60 dB   | 39.45 dB   |  - | __ |
|W2S Ch.1 Avg 16| 38.46 dB   | 38.41 dB   |  - | __ |
|W2S Ch.2 Avg 16| 40.32 dB   | 40.56 dB   |  - | __ |
## Qualitative Results
### DenoiSeg
#### Mouse
![Sample From Test Set](./Results/DenoiSeg_Mouse/DenoiSegMouse.png)
#### Mouse s&p
![Sample From Test Set](./Results/DenoiSeg_Mouse_s&p/DenoiSegMouseS&p.png)
#### Flywing
![Sample From Test Set](./Results/DenoiSeg_Flywing/DenoiSegFlywing.png)

### W2S
#### Ch.0 Avg1
![Sample From Test Set](./Results/W2S/CH0_Avg1.png)
#### Ch.1 Avg1
![Sample From Test Set](./Results/W2S/CH1_Avg1.png)
#### Ch.2 Avg1
![Sample From Test Set](./Results/W2S/CH2_Avg1.png)


## Citation
```bibtex
@article{2020DivNoising,
  title={DivNoising: Diversity Denoising with Fully Cnvolutional Variational Autoencoders},
  author={Prakash, Mangal and Alexander Krull and Jug, Florian},
  journal={arXiv preprint arXiv:2006.06072},
  year={2020}
}
