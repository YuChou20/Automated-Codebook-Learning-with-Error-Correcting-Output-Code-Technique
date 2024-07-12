# Automated Codebook Learning with Error Correcting Output Code Technique

Error Correcting Output Codes (ECOC) is a technique for solving multi-class classification problems. Its core concept involves designing a codebook: each class maps to a unique codeword; these codewords are treated as labels for model training. Thus, the design of the codebook is crucial. In past research, codebooks were often manually designed based on known encoding techniques or generated randomly. However, these methods require manual codebook design before model training, and there may be better choices of codebooks for the given datasets. This research proposes three automated codebook learning models ACL, ACL-CFPC and ACL-TFC for ECOC based on the framework of contrastive learning. These models do not require manual codebook design before training, and the model automatically learns the codebook based on the dataset's characteristics. 

The research also provides two baseline model, Simple and SimCLR for comparison. More detailed content is written in the master's thesis.

## Environment settings
The experimental environment required for this research can be built through the provided dockerfile.
```
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

MAINTAINER yu

WORKDIR /home/

RUN pip install tensorboard
RUN pip install matplotlib

COPY ./ ./
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

### ACL, ACL-CFPC model pre-training
#### Example
After executing run.py, the system will automatically create a folder and save the model's configuration file as well as the weight file. In subsequent model fine-tuning training and testing, this folder must be specified to load the model's weight file and continue training and testiong based on it.
```python
$ python3 run.py --dataset-name cifar10 --epochs 2000 --model_type acl --csl_lambda 0.001 --code_dim 100 --save_weight_every_n_steps 100
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `--data` | `./datasets`   | The location where the datasets used in the code are stored.  |
| `--dataset-name` | `cifar10`   | Dataset name.  <br/> Options: `cifar10`, `mnist`, `fashion-mnist`, `gtsrb`|
| `--lr` | `1.0`   | learning rate used in the model. |
| `--batch_size` | `256`   |  Batch size used in the model. |
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
$ python3 acl_finetune_training.py --folder_name cifar10-simclr-code100 --epochs 200 --pretrain_epochs 1800 
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `--folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `--batch_size` | `256`   |  Batch size used in the model. |
| `--epochs` | `200`   | The number of total epochs to run for model training in finetuning phase. |
| `--pretrain_epochs` | `1800`   | The epochs of the pre-trained model. The system will automatically access the pre-trained weight files using the default naming convention. |

### ACL Model finetuning (Testing)
#### Example

```python
$ python3 acl_finetune_testing.py --folder_name cifar10-simclr-code100 --weight_name acl_best_checkpoint_1.pth.tar --attack_type FGSM
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `--folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `--weight_name` | `acl_best_checkpoint_1.pth.tar`   | The name of the model weights to be evaluated. |
| `--batch_size` | `256`   |  Batch size used in the model. |
| `--attack_type` | `FGSM`   | The method be used to generate adversarial examples. <br> Options: `FGSM`, `PGD`. |
| `--max_iter` | `200`   | Max iteration for PGD attack.  |
| `--epsilon` | `0.031`   | $\epsilon$ in FGSM and PGD attack.  |
| `--eps_step` | `0.01`   | $\alpha$ in PGD attack. |

### ACL-CFPC Model finetuning (Training)
#### Example

```python
$ python3 acl_cfpc_finetune_training.py --folder_name cifar10-simclr-code100 --epochs 200 --pretrain_epochs 1800 --learned_codebook_name cifar10_100bits_codebooks
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `--folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `--batch_size` | `256`   |  Batch size used in the model. |
| `--epochs` | `200`   | The number of total epochs to run for model training in finetuning phase. |
| `--pretrain_epochs` | `1800`   | The epochs of the pre-trained model. The system will automatically access the pre-trained weight files using the default naming convention. |
| `--learned_codebook_name` | `cifar10_100bits_codebooks`   | Learned codebook name generated from ACL-CFPC model. |

### ACL-CFPC Model finetuning (Testing)
#### Example

```python
$ python3 acl_cfpc_finetune_testing.py --folder_name cifar10-simclr-code100 --weight_name acl_cfpc_best_checkpoint_1.pth.tar --attack_type FGSM
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `--folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `--weight_name` | `acl_cfpc_best_checkpoint_1.pth.tar`   | The name of the model weights to be evaluated. |
| `--batch_size` | `256`   |  Batch size used in the model. |
| `--attack_type` | `FGSM`   | The method be used to generate adversarial examples. <br> Options: `FGSM`, `PGD`. |
| `--max_iter` | `200`   | Max iteration for PGD attack.  |
| `--epsilon` | `0.031`   | $\epsilon$ in FGSM and PGD attack.  |
| `--eps_step` | `0.01`   | $\alpha$ in PGD attack. |

### ACL-TFC Model Training
#### Example

```python
$ python3 acl_tfc_training.py --folder_name cifar10-simclr-code100 --dataset_name cifar10 --batch_size 128 --epochs 2000 --learned_codebook cifar10_100bits_codebooks.npy --code_dim 100
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `--folder_name` | `cifar10-simclr-code100`   | The storage location for the model weights thereafter.  |
| `--dataset_name` | `cifar10`   | Dataset name.  <br/> Options: `cifar10`, `mnist`, `fashion-mnist`, `gtsrb`|
| `--batch_size` | `256`   |  Batch size used in the model. |
| `--epochs` | `2000`   | The number of total epochs to run for model training. |
| `--learned_codebook` | `cifar10_100bits_codebooks.npy`   | Learned codebook generated from ACL-CFPC model. |
| `--code_dim` | `100`   | The length of each codeword. |
| `--temperature` | `0.5`   | The temperature used for InfoNCE. |
> [!WARNING]  
> The length of the codewords in learned codebook must match to the architecture of the ACL-TFC model.

### ACL-TFC Model Testing
#### Example

```python
$ python3 acl_tfc_testing.py --folder_name cifar10-simclr-code100 --batch_size 128 --weight_name acl_tfc_best_checkpoint_1.pth.tar --attack_type FGSM 
```

#### Options
| Name      | Default | Description |
|-----------|---------|-------------|
| `--folder_name` | `cifar10-simclr-code100`   | Pre-trained model weights storage location and storage location for fine-tuning model weights thereafter.  |
| `--batch_size` | `256`   |  Batch size used in the model. |
| `--weight_name` | `acl_tfc_best_checkpoint_1.pth.tar`   | The name of the model weights to be evaluated. |
| `--attack_type` | `FGSM`   | The method be used to generate adversarial examples. <br> Options: `FGSM`, `PGD`. |
| `--max_iter` | `200`   | Max iteration for PGD attack.  |
| `--epsilon` | `0.031`   | $\epsilon$ in FGSM and PGD attack.  |
| `--eps_step` | `0.01`   | $\alpha$ in PGD attack. |
