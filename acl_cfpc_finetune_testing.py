import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import argparse
import logging
from models.resnet_ecoc_simclr import ResNetECOCSimCLR
from torch.utils.tensorboard import SummaryWriter
import AdversarialAttackCleverHans_ecoc
import torch.nn.functional as F
import torch.nn as nn
from utils import save_checkpoint
import copy
from torchvision import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
# Model settings
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--folder_name', default='cifar10-simclr-code100',
                    help='model file name')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names, 
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--loss_type', default='CE+HL+RSL',
                    help='model loss type')
parser.add_argument('--weight_name', default='(CE+HL+RSL)acl_cfpc_best_checkpoint_1.pth.tar',
                    help='model weight name')

# Attack PGD
parser.add_argument('--attack_type', default='FGSM',
                    help='FGSM' or 'PGD' or 'CWL2 or None', choices=['FGSM', 'PGD', 'CWL2', 'None'])
parser.add_argument('--norm', default=np.inf , type=int, metavar='N',
                    help='norm of the attack (np.inf or 2)')
parser.add_argument('--max_iter', default=200, type=int, metavar='N',
                    help='max iteration for PGD attack ')
parser.add_argument('--epsilon', default=0.031, type=int, metavar='N',
                    help='bound the attack norm ')
parser.add_argument('--eps_step', default=0.01, type=int, metavar='N',
                    help='epsilon step for PGD attack')
parser.add_argument('--loss', default=torch.nn.CrossEntropyLoss(),
                    help='number of total epochs to run')

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.STL10('./datasets', split='train', download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.STL10('./datasets', split='test', download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.CIFAR10('./datasets', train=True, download=download,
                                  transform=transforms.ToTensor())
  # split train data into data for generate codeword and for training.
  codeword_data_size = int(0.2*len(train_dataset))
  train_size = len(train_dataset) - codeword_data_size
  lengths = [codeword_data_size, train_size]
  codeword_gen_dataset, train_dataset = torch.utils.data.dataset.random_split(train_dataset, lengths)

  codeword_gen_loader = DataLoader(codeword_gen_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.CIFAR10('./datasets', train=False, download=download,
                                  transform=transforms.ToTensor())
  val_data_size = int(0.2*len(test_dataset))
  test_size = len(test_dataset) - val_data_size
  lengths = [val_data_size, test_size]
  val_dataset, test_dataset = torch.utils.data.dataset.random_split(test_dataset, lengths)
  val_loader = DataLoader(val_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return codeword_gen_loader, train_loader, val_loader, test_loader

def get_mnist_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.MNIST('./datasets', train=True, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))
  # split train data into data for generate codeword and for training.
  codeword_data_size = int(0.2*len(train_dataset))
  train_size = len(train_dataset) - codeword_data_size
  lengths = [codeword_data_size, train_size]
  codeword_gen_dataset, train_dataset = torch.utils.data.dataset.random_split(train_dataset, lengths)

  codeword_gen_loader = DataLoader(codeword_gen_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  test_dataset = datasets.MNIST('./datasets', train=False, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return codeword_gen_loader, train_loader, test_loader

def get_fashion_mnist_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.FashionMNIST('./datasets', train=True, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))
  # split train data into data for generate codeword and for training.
  codeword_data_size = int(0.2*len(train_dataset))
  train_size = len(train_dataset) - codeword_data_size
  lengths = [codeword_data_size, train_size]
  codeword_gen_dataset, train_dataset = torch.utils.data.dataset.random_split(train_dataset, lengths)

  codeword_gen_loader = DataLoader(codeword_gen_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  test_dataset = datasets.FashionMNIST('./datasets', train=False, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return codeword_gen_loader, train_loader, test_loader

def get_gtsrb_data_loaders(download, shuffle=True, batch_size=256):
  train_dataset = datasets.GTSRB('./datasets', split ='train', download=download,
                                  transform=transforms.Compose([transforms.RandomResizedCrop(size=16), transforms.ToTensor()]))
  # split train data into data for generate codeword and for training.
  codeword_data_size = int(0.2*len(train_dataset))
  train_size = len(train_dataset) - codeword_data_size
  lengths = [codeword_data_size, train_size]
  codeword_gen_dataset, train_dataset = torch.utils.data.dataset.random_split(train_dataset, lengths)

  codeword_gen_loader = DataLoader(codeword_gen_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  test_dataset = datasets.GTSRB('./datasets', split ='test', download=download,
                                  transform=transforms.Compose([transforms.RandomResizedCrop(size=16), transforms.ToTensor()]))

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return codeword_gen_loader, train_loader, test_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def generate_codeword(model, data, class_num=10, out_dim=2048):
  # Record the codeword for each category
  codewords = np.zeros((class_num, out_dim))
  # Record how many samples there are for each category
  num_of_class = np.zeros(class_num)
  for x, y in data:
    x = x.to(device)
    # Get outputs of ecoc encoder
    logits = model(x)
    for c in range(class_num):
      # Get the sample index belonging to class c
      yi = (y == c).nonzero(as_tuple=True)
      num_of_class[c] += len(yi[0])
      # Sum the current codeword of class c
      codewords[c,:] += torch.sum(logits[yi],dim=0).data.cpu().numpy()
  # Average the codeword
  for i in range(class_num):
    codewords[i] /= num_of_class[i]
  return codewords

def ecoc_decodering(features, codewords):
    codewords = torch.tensor(codewords, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
    # normalize features and codewords by their L2 norm
    features = features / torch.norm(features, dim=1, keepdim=True)
    codewords = codewords / torch.norm(codewords, dim=1, keepdim=True)
    softmax = nn.Softmax(dim=1)
    return softmax(torch.einsum('ij,kj->ik', features, codewords))

def get_hinge_loss(features, codewords, labels):
    loss = 0
    for i in range(features.shape[0]):
      cos = torch.nn.CosineSimilarity(dim=0)
      y = torch.tensor(codewords[labels[i]], dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
      loss += max(0.5-cos(features[i], y),0)
    return torch.tensor(loss, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)

def get_attacked_dataset(attack_type, model, test_data, codewords, eps):
  if attack_type == 'FGSM':
      data_perturbed = AdversarialAttackCleverHans_ecoc.FGSM(model, config.batch_size, test_data, eps, args.norm, codewords=codewords)             
  elif attack_type == 'PGD':
      data_perturbed = AdversarialAttackCleverHans_ecoc.PGD(model, config.batch_size, test_data, eps, eps/3, args.max_iter, args.norm, args.loss, early_stop=True, codewords=codewords)
  elif attack_type == 'CWL2':
      data_perturbed = AdversarialAttackCleverHans_ecoc.CW_L2(model, config.batch_size, test_data, max_iter=args.max_iter, mode="ECOC-SimCLR", codewords=codewords)
  return data_perturbed

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Using device:", device)
  args = parser.parse_args()
  writer = SummaryWriter()
  # Load config.yml
  with open(os.path.join('./runs/{0}/config.yml'.format(args.folder_name))) as file:
    config = yaml.load(file, Loader=yaml.Loader)

  code_dim = config.code_dim
  class_num = 43 if config.dataset_name == 'gtsrb' else 10
  # Get baseline model arch
  if config.arch == 'resnet18':
    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
  elif config.arch == 'resnet50':
    model = ResNetECOCSimCLR(base_model=config.arch, out_dim=class_num, code_dim=code_dim)
    
    # Remove last relu
    model.ecoc_encoder[1]= nn.Identity()
    model.fc = nn.Identity()
  print(model)


  model.cuda()
  # print(model)
  # Load weight file
  checkpoint = torch.load('./runs/{0}/{1}'.format(args.folder_name, args.weight_name), map_location=device)
  state_dict = checkpoint['state_dict']
  state_dict_cpy = state_dict.copy()

  log = model.load_state_dict(state_dict, strict=False)
  # assert log.missing_keys == []

  # Load dataset to loader
  if config.dataset_name == 'cifar10':
    codeword_gen_loader, train_loader, val_loader, test_loader = get_cifar10_data_loaders(download=True,batch_size=args.batch_size)
  elif config.dataset_name == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=True,batch_size=args.batch_size)
  elif config.dataset_name == 'mnist':
    codeword_gen_loader, train_loader, test_loader = get_mnist_data_loaders(download=True,batch_size=args.batch_size)
  elif config.dataset_name == 'fashion-mnist':
    codeword_gen_loader, train_loader, test_loader = get_fashion_mnist_data_loaders(download=True,batch_size=args.batch_size)
  elif config.dataset_name == 'gtsrb':
    codeword_gen_loader, train_loader, test_loader = get_gtsrb_data_loaders(download=True,batch_size=args.batch_size)
  print("Dataset:", config.dataset_name)
  

  requires_grad_list = ['ecoc_encoder.0.weight', 'ecoc_encoder.0.bias']

  # freeze all layers but the last fc
  for name, param in model.named_parameters():
      # print(name, param)
      if name not in requires_grad_list:
          param.requires_grad = False
      else:
          param.requires_grad = True
  parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
  assert len(parameters) == len(requires_grad_list)

  # Assign some model settings
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
  criterion = torch.nn.CrossEntropyLoss().to(device)

  # training & testing
  logging.basicConfig(filename=os.path.join('./runs/{0}'.format(args.folder_name), '({0})acl_cpfc_{1}.log'.format(args.loss_type,args.attack_type)), level=logging.DEBUG)
  logging.info(args.weight_name)
  print(args.weight_name)
  if args.attack_type != 'None':
    logging.info('attack_type = {0}'.format(args.attack_type))
    logging.info('norm = {0}'.format(args.norm))
    if args.attack_type != 'FGSM':
      logging.info('max_iter = {0}'.format(args.max_iter))
    if args.attack_type != 'CWL2':
      logging.info('epsilon = {0}'.format(args.epsilon))
    if args.attack_type == 'PGD':
      logging.info('eps_step = {0}'.format(args.eps_step))
  codewords = generate_codeword(model, codeword_gen_loader, class_num=class_num, out_dim=code_dim)

  logging.info('eps= {0}'.format(args.epsilon))
  top1_accuracy = 0
  top5_accuracy = 0
  test_acc = []
  attacked_top1_accuracy = 0
  attacked_top5_accuracy = 0
  att_acc = []

  codewords_test = generate_codeword(model, codeword_gen_loader, class_num=class_num, out_dim=code_dim)
  # testing
  for counter, (x_batch, y_batch) in enumerate(test_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.type(torch.LongTensor).to(device)

    pred_class = ecoc_decodering(model(x_batch), codewords_test)

    top1, top5 = accuracy(pred_class, y_batch, topk=(1,5))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]

  top1_accuracy /= (counter + 1)
  top5_accuracy /= (counter + 1)
  test_acc.append(top1_accuracy.cpu())
  print(f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
  logging.info(f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
  # Get attacked dataset if attack
  if args.attack_type != 'None':
    perturbed_test_loader = get_attacked_dataset(attack_type=args.attack_type, model=model, test_data=test_loader, codewords=codewords_test, eps=args.epsilon)
    for counter, (x_batch, y_batch) in enumerate(perturbed_test_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.type(torch.LongTensor).to(device)

      pred_class = ecoc_decodering(model(x_batch), codewords_test)

      top1, top5 = accuracy(pred_class, y_batch, topk=(1,5))
      attacked_top1_accuracy += top1[0]
      attacked_top5_accuracy += top5[0]

    attacked_top1_accuracy /= (counter + 1)
    attacked_top5_accuracy /= (counter + 1)
    print(f"Attacked Top1 Test accuracy: {attacked_top1_accuracy.item()}\tTop5 test acc: {attacked_top5_accuracy.item()}")
    logging.info(f"Attacked Top1 Test accuracy: {attacked_top1_accuracy.item()}\tTop5 test acc: {attacked_top5_accuracy.item()}")
