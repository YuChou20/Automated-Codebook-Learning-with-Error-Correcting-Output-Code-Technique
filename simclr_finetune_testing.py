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
import AdversarialAttackCleverHans
from models.resnet_ecoc_simclr import ResNetECOCSimCLR
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint
import copy
from torchvision import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

# Model settings
parser.add_argument('-folder_name', default='cifar10-simclr-code100',
                    help='model file name')
parser.add_argument('--pretrain_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-weight_name', default='(simclr_finetune(CE+HL+RSL)cifar10-simclr-code100_best_checkpoint_1.pth.tar',
                    help='model weight name')

# Attack settings
parser.add_argument('--attack_type', default='FGSM',
                    help='FGSM or PGD or CWL2 or None', choices=['FGSM', 'PGD', 'CWL2', 'None'])
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
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, val_loader, test_loader

def get_mnist_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.MNIST('./datasets', train=True, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.MNIST('./datasets', train=False, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))
  val_data_size = int(0.2*len(test_dataset))
  test_size = len(test_dataset) - val_data_size
  lengths = [val_data_size, test_size]
  val_dataset, test_dataset = torch.utils.data.dataset.random_split(test_dataset, lengths)
  val_loader = DataLoader(val_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, val_loader, test_loader

def get_fashion_mnist_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.FashionMNIST('./datasets', train=True, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.FashionMNIST('./datasets', train=False, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))
  val_data_size = int(0.2*len(test_dataset))
  test_size = len(test_dataset) - val_data_size
  lengths = [val_data_size, test_size]
  val_dataset, test_dataset = torch.utils.data.dataset.random_split(test_dataset, lengths)
  val_loader = DataLoader(val_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, val_loader, test_loader

def get_gtsrb_data_loaders(download, shuffle=True, batch_size=256):
  train_dataset = datasets.GTSRB('./datasets', split ='train', download=download,
                                  transform=transforms.Compose([transforms.RandomResizedCrop(size=16), transforms.ToTensor()]))

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.GTSRB('./datasets', split ='test', download=download,
                                  transform=transforms.Compose([transforms.RandomResizedCrop(size=16), transforms.ToTensor()]))
  val_data_size = int(0.2*len(test_dataset))
  test_size = len(test_dataset) - val_data_size
  lengths = [val_data_size, test_size]
  val_dataset, test_dataset = torch.utils.data.dataset.random_split(test_dataset, lengths)
  val_loader = DataLoader(val_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, val_loader, test_loader

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

def get_attacked_dataset(attack_type, model, test_data):
  data_perturbed = {}
  if attack_type == 'FGSM':
      data_perturbed = AdversarialAttackCleverHans.FGSM(model, config.batch_size, test_data, args.epsilon, args.norm)             
  elif attack_type == 'PGD':
      data_perturbed = AdversarialAttackCleverHans.PGD(model, config.batch_size, test_data, args.epsilon, args.eps_step, args.max_iter, args.norm, args.loss, early_stop=True)
  elif attack_type == 'CWL2':
      data_perturbed = AdversarialAttackCleverHans.CW_L2(model, config.batch_size, test_data, max_iter=args.max_iter)
  return data_perturbed

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Using device:", device)
  args = parser.parse_args()
  writer = SummaryWriter()
  # Load config.yml
  with open(os.path.join('./runs/{0}/config.yml'.format(args.folder_name))) as file:
    config = yaml.load(file, Loader=yaml.Loader)

  # cp_epoch = (len(str(config.epochs)))*'0' + str(config.epochs)
  cp_epoch = '{:04d}'.format(args.pretrain_epochs)
  code_dim = config.code_dim
  class_num = 43 if config.dataset_name == 'gtsrb' else 10
  # Get baseline model arch
  if config.arch == 'resnet18':
    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
  elif config.arch == 'resnet50':
    model = ResNetECOCSimCLR(base_model=config.arch, out_dim=class_num, code_dim=code_dim)
    dim_mlp = model.ecoc_encoder[0].out_features
    model.fc = nn.Linear(dim_mlp, class_num)
    print(model)

  model.cuda()
  # Load weight file
  checkpoint = torch.load('./runs/{0}/{1}'.format(args.folder_name, args.weight_name), map_location=device)
  state_dict = checkpoint['state_dict']
  state_dict_cpy = state_dict.copy()


  log = model.load_state_dict(state_dict, strict=False)
  # assert log.missing_keys == ['fc.weight', 'fc.bias']

  # Load dataset to loader
  if config.dataset_name == 'cifar10':
    train_loader, val_loader, test_loader = get_cifar10_data_loaders(download=True)
  elif config.dataset_name == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=True)
  elif config.dataset_name == 'mnist':
    train_loader, val_loader, test_loader = get_mnist_data_loaders(download=True)
  elif config.dataset_name == 'fashion-mnist':
    train_loader, val_loader, test_loader = get_fashion_mnist_data_loaders(download=True)
  elif config.dataset_name == 'gtsrb':
    train_loader, val_loader, test_loader = get_gtsrb_data_loaders(download=True)
  print("Dataset:", config.dataset_name)

  requires_grad_list = ['ecoc_encoder.0.weight', 'ecoc_encoder.0.bias', 'fc.weight', 'fc.bias']

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

  logging.basicConfig(filename=os.path.join('./runs/{0}'.format(args.folder_name), '({0})simclr_finetune_testing.log'.format(cp_epoch)), level=logging.DEBUG)
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

  top1_accuracy = 0
  top5_accuracy = 0
  attacked_top1_accuracy = 0
  attacked_top5_accuracy = 0

  # testing
  for counter, (x_batch, y_batch) in enumerate(test_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.type(torch.LongTensor).to(device)

    logits = model(x_batch)

    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]

  top1_accuracy /= (counter + 1)
  top5_accuracy /= (counter + 1)
  print(f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
  logging.info(f"\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
  # Get attacked dataset if attack
  if args.attack_type != 'None':
    perturbed_test_loader = get_attacked_dataset(attack_type=args.attack_type, model=model, test_data=test_loader)
    for counter, (x_batch, y_batch) in enumerate(perturbed_test_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.type(torch.LongTensor).to(device)
      logits = model(x_batch)

      top1, top5 = accuracy(logits, y_batch, topk=(1,5))
      attacked_top1_accuracy += top1[0]
      attacked_top5_accuracy += top5[0]

    attacked_top1_accuracy /= (counter + 1)
    attacked_top5_accuracy /= (counter + 1)
    print(f"Attacked Top1 Test accuracy: {attacked_top1_accuracy.item()}\tTop5 test acc: {attacked_top5_accuracy.item()}")
    logging.info(f"Attacked Top1 Test accuracy: {attacked_top1_accuracy.item()}\tTop5 test acc: {attacked_top5_accuracy.item()}")