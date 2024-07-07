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
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

def get_stl10_data_loaders(download, shuffle=False, batch_size=128):
  train_dataset = datasets.STL10('./datasets', split='train', download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.STL10('./datasets', split='test', download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=128):
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

def get_mnist_data_loaders(download, shuffle=False, batch_size=128):
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

def get_fashion_mnist_data_loaders(download, shuffle=False, batch_size=128):
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

def get_gtsrb_data_loaders(download, shuffle=True, batch_size=128):
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
    dim_mlp = model.ecoc_encoder[0].out_features
    model.fc = nn.Linear(dim_mlp, class_num)
    print(model)

  model.cuda()

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
      param.requires_grad = True
  parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
  # assert len(parameters) == len(requires_grad_list)

  # Assign some model settings
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
  criterion = torch.nn.CrossEntropyLoss().to(device)

  # training & testing
  logging.basicConfig(filename=os.path.join('./runs/{0}'.format(args.folder_name), 'simple_training.log'), level=logging.DEBUG)
  logging.info('{0}'.format(args.folder_name))

  best_val_acc = 0
  test_acc = 0
  best_epoch = 0
  best_model = copy.deepcopy(model.state_dict())
  best_optimizer = copy.deepcopy(optimizer.state_dict())
  for epoch in range(1, args.epochs+1):
    top1_train_accuracy = 0
    # training
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
    val_top1_accuracy = 0
    val_top5_accuracy = 0
    # valiadating
    for counter, (x_batch, y_batch) in enumerate(val_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.type(torch.LongTensor).to(device)

      logits = model(x_batch)

      top1, top5 = accuracy(logits, y_batch, topk=(1,5))
      val_top1_accuracy += top1[0]
      val_top5_accuracy += top5[0]

    val_top1_accuracy /= (counter + 1)
    val_top5_accuracy /= (counter + 1)

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
    if torch.ge(val_top1_accuracy, best_val_acc, out=None):
      best_val_acc = val_top1_accuracy
      test_acc = top1_accuracy
      best_epoch = epoch
      best_model = copy.deepcopy(model.state_dict())
      best_optimizer = copy.deepcopy(optimizer.state_dict())
    writer.add_scalar('loss', loss, global_step=epoch)
    writer.add_scalar('Eval Train: acc/top1', top1_train_accuracy, global_step=epoch)
    writer.add_scalar('Eval: val acc/top1', val_top1_accuracy, global_step=epoch)
    writer.add_scalar('Eval: val acc/top5', val_top5_accuracy, global_step=epoch)
    writer.add_scalar('Eval: acc/top1', top1_accuracy, global_step=epoch)
    writer.add_scalar('Eval: acc/top5', top5_accuracy, global_step=epoch)
    logging.info(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 val accuracy: {val_top1_accuracy.item()}\tTop5 val acc: {val_top5_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 val accuracy: {val_top1_accuracy.item()}\tTop5 val acc: {val_top5_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    #
  checkpoint_name = 'simple_best_checkpoint_{0}.pth.tar'.format(best_epoch)
  save_checkpoint({
      'epoch': best_epoch,
      'arch': args.arch,
      'state_dict': best_model,
      'optimizer': best_optimizer,
  }, is_best=False, filename='./runs/{0}/{1}'.format(args.folder_name, checkpoint_name))
  logging.info("best_epoch = {0}, best_val_acc = {1}, test_acc = {2}".format(best_epoch, best_val_acc, test_acc))
  print("{0} best_epoch = {1}, best_val_acc = {2}, test_acc = {3}".format(args.folder_name, best_epoch, best_val_acc, test_acc))