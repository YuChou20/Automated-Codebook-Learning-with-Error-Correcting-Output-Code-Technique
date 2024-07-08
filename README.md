# Automated Codebook Learning with Error Correcting Output Code Technique

Error Correcting Output Codes (ECOC) is a technique for solving multi-class classification problems. Its core concept involves designing a codebook: each class maps to a unique codeword; these codewords are treated as labels for model training. Thus, the design of the codebook is crucial. In past research, codebooks were often manually designed based on known encoding techniques or generated randomly. However, these methods require manual codebook design before model training, and there may be better choices of codebooks for the given datasets. This research proposes three automated codebook learning models ACL, ACL-CFPC and ACL-TFC for ECOC based on the framework of contrastive learning. These models do not require manual codebook design before training, and the model automatically learns the codebook based on the dataset's characteristics. 

The research also provides two baseline model, Simple and SimCLR for comparison. More detailed content is written in the master's thesis.
## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Main File Description
* `run.py`: This program is used for the pre-training phase of SimCLR, ACL and ACL-CFPC models.
* `simclr.py`: The definitions related to the pre-trained model.
* `simple_training.py`: The file used for training the Simple model. This model does not have a pre-training phase and can directly proceed to model training.
* `simple_testing.py`: This file is used to evaluate the performance of the Simple model. Model training must be completed using `simple_training.py` to obtain the trained model weight before evaluating its performance.
* `simclr_finetune_training.py`: The file used for training the SimCLR model in the finetuning phase. Before starting to finetune the model, you must complete the pre-training phase using `run.py`.
* `simclr_finetune_testing.py`: This file is used to evaluate the performance of the SimCLR model. Model training must be completed using `simclr_finetune_testing.py` to obtain the trained model weight before evaluating its performance.
* `acl_finetune_training.py`: The file used for training the ACL model in the finetuning phase. Before starting to finetune the model, you must complete the pre-training phase using `run.py`.
* `acl_finetune_testing.py`: This file is used to evaluate the performance of the ACL model. Model training must be completed using `acl_finetune_training.py` to obtain the trained model weight before evaluating its performance.
* `acl_cfpc_finetune_training.py`: The file used for training the ACL-CFPC model in the finetuning phase. Before starting to finetune the model, you must complete the pre-training phase using `run.py`.
* `acl_cfpc_finetune_testing.py`: This file is used to evaluate the performance of the ACL-CFPC model. Model training must be completed using `acl_cfpc_finetune_training.py` to obtain the trained model weight before evaluating its performance.
* `acl_tfc_training.py`: The file used for training the ACL-TFC model. This model does not have a pre-training phase, but it requires the ACL-CFPC model to be trained first to obtain the learned codebook before training the ACL-TFC model.
* `acl_tfc_testing.py`: This file is used to evaluate the performance of the ACL-TFC model. Model training must be completed using `acl_tfc_training.py` to obtain the trained model weight before evaluating its performance.


## Datasets
This study utilizes four datasets: [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10), [MNIST](https://www.tensorflow.org/datasets/catalog/mnist), [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist), and [GTSRB](https://www.tensorflow.org/datasets/catalog/visual_domain_decathlon). There is no need to manually download these datasets before running the program; all datasets will be automatically downloaded during program execution. The default path for the datasets is "./datasets". Please create the corresponding folder before running any programs.

## Quick Start
Quick Start provides example commands for training different models and lists some of the more important options. For additional settings, you can adjust the experimental configurations according to the available options in the options section of the code.

### ACL model pre-training
#### Example

```python
$ python run.py -dataset-name cifar10 --epochs 2000 --model_type acl --csl_lambda 0.001 --code_dim 100 --save_weight_every_n_steps 100
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `-data` | `./datasets`   | The location where the datasets used in the code are stored.  |
| `-dataset-name` | `cifar10`   | Dataset name.  <br/> Options: `cifar10`, `mnist`, `fashion-mnist`, `gtsrb`|
| `--lr` | `1.0`   | learning rate used in the model. |
| `--log-every-n-steps` | `1`   | Log the infomation every n steps. |
| `--temperature` | `0.5`   | The temperature used for InfoNCE. |
| `--epochs` | `2000`   | The number of total epochs to run for model training. |
| `--model_type` | `acl`   | Choose the model for model pre-training. <br/> Options: `simclr`, `acl`. |
| `--csl_lambda` | `0.001`   | The parameter used for controling the weight of column separation loss. Only used when `--model_type` set as `acl`. |
| `--code_dim` | `100`   | The length of each codeword. |
| `--save_weight_every_n_steps` | `100`   |Save weights every n epochs. |

### ACL Model finetuning (Training)
#### Example

```python
$ python acl_finetune_training.py -folder_name cifar10-simclr-code100 --epochs 200 --pretrain_epochs 1800 
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `-folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `--epochs` | `200`   | The number of total epochs to run for model training in finetuning phase. |

### ACL Model finetuning (Testing)
#### Example

```python
$ python acl_finetune_testing.py -folder_name cifar10-simclr-code100 --attack_type FGSM 
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `-folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `-weight_name` | `(CE+HL+RSL)acl_best_checkpoint_1.pth.tar`   | The name of the model weights to be evaluated. |
| `--max_iter` | `200`   | Max iteration for PGD attack.  |
| `--epsilon` | `0.031`   | $\epsilon$ in FGSM and PGD attack.  |
| `--eps_step` | `0.01`   | $\alpha$ in PGD attack. |



### ACL-CFPC Model finetuning (Training)
#### Example

```python
$ python acl_cfpc_finetune_training.py -folder_name cifar10-simclr-code100 --epochs 200 --pretrain_epochs 1800 -learned_codebook_name (CE+HL+RSL)cifar10_100bits_codebooks
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `-folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `--epochs` | `200`   | The number of total epochs to run for model training in finetuning phase. |
| `-learned_codebook_name` | `(CE+HL+RSL)cifar10_100bits_codebooks`   | Learned codebook name generated from ACL-CFPC model. |

### ACL-TFC Model Training
#### Example

```python
$ python acl_tfc_training.py -folder_name cifar10-simclr-code100 -dataset_name cifar10 --epochs 2000 -learned_codebook (CE+HL+RSL)cifar10_100bits_codebooks.npy --code_dim 100
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `-folder_name` | `cifar10-simclr-code100`   | The storage location for the model weights thereafter.  |
| `-dataset_name` | `cifar10`   | Dataset name.  <br/> Options: `cifar10`, `mnist`, `fashion-mnist`, `gtsrb`|
| `--epochs` | `2000`   | The number of total epochs to run for model training. |
| `-learned_codebook` | `(CE+HL+RSL)cifar10_100bits_codebooks.npy`   | Learned codebook generated from ACL-CFPC model. |
| `--code_dim` | `100`   | The length of each codeword. |
| `--temperature` | `0.5`   | The temperature used for InfoNCE. |
> [!WARNING]  
> The length of the codewords in learned codebook must match to the arch of the ACL-TFC model.



## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.

For 16-bit precision GPU training, there **NO** need to to install [NVIDIA apex](https://github.com/NVIDIA/apex). Just use the ```--fp16_precision``` flag and this implementation will use [Pytorch built in AMP training](https://pytorch.org/docs/stable/notes/amp_examples.html).

## Feature Evaluation

Feature evaluation is done using a linear model protocol. 

First, we learned features using SimCLR on the ```STL10 unsupervised``` set. Then, we train a linear classifier on top of the frozen features from SimCLR. The linear model is trained on features extracted from the ```STL10 train``` set and evaluated on the ```STL10 test``` set. 

Check the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/sthalles/SimCLR/blob/simclr-refactor/feature_eval/mini_batch_logistic_regression_evaluator.ipynb) notebook for reproducibility.

Note that SimCLR benefits from **longer training**.

| Linear Classification      | Dataset | Feature Extractor | Architecture                                                                    | Feature dimensionality | Projection Head dimensionality | Epochs | Top1 % |
|----------------------------|---------|-------------------|---------------------------------------------------------------------------------|------------------------|--------------------------------|--------|--------|
| Logistic Regression (Adam) | STL10   | SimCLR            | [ResNet-18](https://drive.google.com/open?id=14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF) | 512                    | 128                            | 100    | 74.45  |
| Logistic Regression (Adam) | CIFAR10 | SimCLR            | [ResNet-18](https://drive.google.com/open?id=1lc2aoVtrAetGn0PnTkOyFzPCIucOJq7C) | 512                    | 128                            | 100    | 69.82  |
| Logistic Regression (Adam) | STL10   | SimCLR            | [ResNet-50](https://drive.google.com/open?id=1ByTKAUsdm_X7tLcii6oAEl5qFRqRMZSu) | 2048                   | 128                            | 50     | 70.075 |
