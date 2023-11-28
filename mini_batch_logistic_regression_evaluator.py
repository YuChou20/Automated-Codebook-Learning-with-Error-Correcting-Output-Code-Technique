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

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder_name', default='cifar10-200-lars-v4-8',
                    help='model file name')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
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

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

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
    
if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Using device:", device)
  args = parser.parse_args()
  with open(os.path.join('./runs/{0}/config.yml'.format(args.folder_name))) as file:
    config = yaml.load(file, Loader=yaml.Loader)

  cp_epoch = (4-len(str(config.epochs)))*'0' + str(config.epochs)

  # Get baseline model arch
  if config.arch == 'resnet18':
    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
  elif config.arch == 'resnet50':
    model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
    dim_mlp = model.fc.in_features
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    if config.model_version == 5:
      model.ecoc_encoder = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU())
      model.fc = nn.Identity(nn.Linear(dim_mlp, dim_mlp))
      model.c_head = nn.Linear(dim_mlp, 10)

  model.cuda()

  checkpoint = torch.load('./runs/{0}/checkpoint_{1}.pth.tar'.format(args.folder_name, cp_epoch), map_location=device)
  state_dict = checkpoint['state_dict']
  state_dict_cpy = state_dict.copy()

  # remove prefix
  if config.model_version == 5:
    for k in list(state_dict.keys()):
      if k.startswith('backbone'):
        state_dict[k[len("backbone."):]] = state_dict[k]
      del state_dict[k]

    state_dict['ecoc_encoder.0.weight'] = state_dict_cpy['ecoc_encoder.0.weight']
    state_dict['ecoc_encoder.0.bias'] = state_dict_cpy['ecoc_encoder.0.bias']

  else:
    for k in list(state_dict.keys()):
      if k.startswith('backbone.'):
        if k.startswith('backbone') and not k.startswith('backbone.fc'):
          # remove prefix
          state_dict[k[len("backbone."):]] = state_dict[k]
      del state_dict[k]

  log = model.load_state_dict(state_dict, strict=False)
  if config.model_version == 5:
    assert log.missing_keys == ['c_head.weight', 'c_head.bias']
  else:
    assert log.missing_keys == ['fc.weight', 'fc.bias']

  if config.dataset_name == 'cifar10':
    train_loader, test_loader = get_cifar10_data_loaders(download=True)
  elif config.dataset_name == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=True)
  print("Dataset:", config.dataset_name)

  requires_grad_list = ['ecoc_encoder.0.weight', 'ecoc_encoder.0.bias', 'c_head.weight', 'c_head.bias'] if config.model_version == 5 else ['fc.weight', 'fc.bias']

  # freeze all layers but the last fc
  for name, param in model.named_parameters():
      # print(name, param)
      if name not in requires_grad_list:
          param.requires_grad = False
      else:
          param.requires_grad = True
  parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
  assert len(parameters) == len(requires_grad_list)  # fc.weight, fc.bias

  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
  criterion = torch.nn.CrossEntropyLoss().to(device)

  # training & testing
  logging.basicConfig(filename=os.path.join('./runs/{0}'.format(args.folder_name), 'eval.log'), level=logging.DEBUG)
  for epoch in range(args.epochs):
    top1_train_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(train_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)

      logits = model(x_batch)
      loss = criterion(logits, y_batch)
      top1 = accuracy(logits, y_batch, topk=(1,))
      top1_train_accuracy += top1[0]

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)

      logits = model(x_batch)

      top1, top5 = accuracy(logits, y_batch, topk=(1,5))
      top1_accuracy += top1[0]
      top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)
    logging.info(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

