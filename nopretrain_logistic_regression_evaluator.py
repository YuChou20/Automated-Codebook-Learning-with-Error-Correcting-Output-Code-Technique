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
import AdversarialAttackCleverHans

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('-folder_name', default='cifar10-lars-v5.2-out100',
                    help='model file name')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--pretrain_epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')

# Attack settings
parser.add_argument('--attack_type', default='None',
                    help='FGSM or PGD or CWL2 or None', choices=['FGSM', 'PGD', 'CWL2', 'None'])
parser.add_argument('--norm', default=np.inf , type=int, metavar='N',
                    help='norm of the attack (np.inf or 2)')
parser.add_argument('--max_iter', default=100, type=int, metavar='N',
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

def get_cifar10_data_loaders(download, shuffle=False, batch_size=128):
  train_dataset = datasets.CIFAR10('./datasets', train=True, download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.CIFAR10('./datasets', train=False, download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

def get_mnist_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.MNIST('./datasets', train=True, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

  test_dataset = datasets.MNIST('./datasets', train=False, download=download,
                                  transform=transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()]))

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
    
def get_attacked_dataset(attack_type, model, test_data):
  if attack_type == 'FGSM':
      data_perturbed = AdversarialAttackCleverHans.FGSM(model, config.batch_size, test_data, args.epsilon, args.norm)             
  elif attack_type == 'PGD':
      data_perturbed = AdversarialAttackCleverHans.PGD(model, config.batch_size, test_data, args.epsilon, args.eps_step, args.max_iter, args.norm, args.loss, early_stop=True)
      print('end*-------------------------')
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
  code_dim = 10#config.code_dim
  
  # Get baseline model arch
  if config.arch == 'resnet18':
    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
  elif config.arch == 'resnet50':
    if config.model_version == 5:
      model = ResNetECOCSimCLR(base_model=config.arch, out_dim=10, code_dim=code_dim)
      dim_mlp = model.ecoc_encoder[0].out_features
      model.fc = nn.Linear(dim_mlp, 10)
    else:
      model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
      dim_mlp = model.fc.in_features
      model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
      model.maxpool = nn.Identity()
    print(model)

  model.cuda()
  # print(model)
  # Load weight file
  # checkpoint = torch.load('./runs/{0}/checkpoint_{1}.pth.tar'.format(args.folder_name, cp_epoch), map_location=device)
  # state_dict = checkpoint['state_dict']
  # state_dict_cpy = state_dict.copy()

  # remove prefix
  # if config.model_version == 5:
  #   for k in list(state_dict.keys()):
  #     if k.startswith('fc.'):
  #       del state_dict[k]
  # else:
  #   for k in list(state_dict.keys()):
  #     if k.startswith('backbone.'):
  #       if k.startswith('backbone') and not k.startswith('backbone.fc'):
  #         # remove prefix
  #         state_dict[k[len("backbone."):]] = state_dict[k]
  #     del state_dict[k]

  # log = model.load_state_dict(state_dict, strict=False)
  # assert log.missing_keys == ['fc.weight', 'fc.bias']

  # Load dataset to loader
  if config.dataset_name == 'cifar10':
    train_loader, test_loader = get_cifar10_data_loaders(download=True)
  elif config.dataset_name == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=True)
  elif config.dataset_name == 'mnist':
    train_loader, test_loader = get_mnist_data_loaders(download=True)
  print("Dataset:", config.dataset_name)

  requires_grad_list = ['ecoc_encoder.0.weight', 'ecoc_encoder.0.bias', 'fc.weight', 'fc.bias'] if config.model_version == 5 else ['fc.weight', 'fc.bias']

  # freeze all layers but the last fc
  for name, param in model.named_parameters():
      # print(name, param)
      if name not in requires_grad_list:
          param.requires_grad = True
      else:
          param.requires_grad = True
  parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
  # assert len(parameters) == len(requires_grad_list)

  # Assign some model settings
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
  criterion = torch.nn.CrossEntropyLoss().to(device)

  # training & testing
  logging.basicConfig(filename=os.path.join('./runs/{0}'.format(args.folder_name), 'eval_nopretrain_{0}.log'.format(args.epochs)), level=logging.DEBUG)
  for epoch in range(args.epochs):
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
    # testing
    if args.attack_type != 'None':
       perturbed_test_loader = get_attacked_dataset(attack_type=args.attack_type, model=model, test_data=test_loader)
    else:
       perturbed_test_loader = test_loader
    for counter, (x_batch, y_batch) in enumerate(perturbed_test_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)

      logits = model(x_batch)

      top1, top5 = accuracy(logits, y_batch, topk=(1,5))
      top1_accuracy += top1[0]
      top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)

    writer.add_scalar('loss', loss, global_step=epoch)
    writer.add_scalar('Eval Train: acc/top1', top1_train_accuracy, global_step=epoch)
    writer.add_scalar('Eval: acc/top1', top1_accuracy, global_step=epoch)
    writer.add_scalar('Eval: acc/top5', top5_accuracy, global_step=epoch)
    
    logging.info(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")