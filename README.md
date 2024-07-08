# Automated Codebook Learning with Error Correcting Output Code Technique

Error Correcting Output Codes (ECOC) is a technique for solving multi-class classification problems. Its core concept involves designing a codebook: each class maps to a unique codeword; these codewords are treated as labels for model training. Thus, the design of the codebook is crucial. In past research, codebooks were often manually designed based on known encoding techniques or generated randomly. However, these methods require manual codebook design before model training, and there may be better choices of codebooks for the given datasets. This research proposes three automated codebook learning models ACL, ACL-CFPC and ACL-TFC for ECOC based on the framework of contrastive learning. These models do not require manual codebook design before training, and the model automatically learns the codebook based on the dataset's characteristics. We compare these models with two baseline models on four open datasets and evaluate the strengths, weaknesses, and limitations of the three ECOC models. Additionally, we experiment with whether the ECOC models with automated codebook learning can resist adversarial attacks and discuss directions for future improvements.

The detailed content is written in the master's thesis, and the complete source code is published in this [project](https://github.com/YuChou20/Automated-Codebook-Learning-with-Error-Correcting-Output-Code-Technique).
## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

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
