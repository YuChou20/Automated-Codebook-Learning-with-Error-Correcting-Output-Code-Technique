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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--pretrain_epochs', default=1800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--loss_type', default='CE+HL+RSL',
                    help='model loss type')
parser.add_argument('--learned_codebook_name', default='(CE+HL+RSL)cifar10_100bits_codebooks',
                    help='Learned codebook name generated from ACL-CFPC model.')

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
  val_data_size = int(0.2*len(test_dataset))
  test_size = len(test_dataset) - val_data_size
  lengths = [val_data_size, test_size]
  val_dataset, test_dataset = torch.utils.data.dataset.random_split(test_dataset, lengths)
  val_loader = DataLoader(val_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return codeword_gen_loader, train_loader, val_loader, test_loader

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
  val_data_size = int(0.2*len(test_dataset))
  test_size = len(test_dataset) - val_data_size
  lengths = [val_data_size, test_size]
  val_dataset, test_dataset = torch.utils.data.dataset.random_split(test_dataset, lengths)
  val_loader = DataLoader(val_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return codeword_gen_loader, train_loader, val_loader, test_loader

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
  val_data_size = int(0.2*len(test_dataset))
  test_size = len(test_dataset) - val_data_size
  lengths = [val_data_size, test_size]
  val_dataset, test_dataset = torch.utils.data.dataset.random_split(test_dataset, lengths)
  val_loader = DataLoader(val_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return codeword_gen_loader, train_loader, val_loader, test_loader

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
 

def generate_codeword_torch(model, data, class_num=10, out_dim=2048):
    # Record the codeword for each category
    codebook = torch.zeros((class_num, out_dim), device=torch.device('cuda:0'))
    # Record how many samples there are for each category
    num_of_class = torch.zeros(class_num, device=torch.device('cuda:0'))
    
    for x, y in data:
        x = x.to(torch.device('cuda:0'))
        # Get outputs of ecoc encoder
        logits = model(x)
        for c in range(class_num):
            # Get the sample index belonging to class c
            yi = (y == c).nonzero(as_tuple=True)
            num_of_samples = len(yi[0])
            num_of_class[c] += num_of_samples
            if num_of_samples > 0:
                # Sum the current codeword of class c
                codebook[c,:] += torch.sum(logits[yi], dim=0)
    
    # Average the codeword
    for i in range(class_num):
        if num_of_class[i] > 0:
            codebook[i] /= num_of_class[i]

    return torch.tensor(codebook, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)

def ecoc_decodering(features, codebook):
    # normalize features and codebook by their L2 norm
    features = features / torch.norm(features, dim=1, keepdim=True)
    codebook = codebook / torch.norm(codebook, dim=1, keepdim=True)
    softmax = nn.Softmax(dim=1)
    return softmax(torch.einsum('ij,kj->ik', features, codebook))

def get_hinge_loss(features, codebook, labels):
    loss = 0
    for i in range(features.shape[0]):
      cos = torch.nn.CosineSimilarity(dim=0)
      y = torch.tensor(codebook[labels[i]], dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
      loss += max(0.5-cos(features[i], y),0)
    return torch.tensor(loss, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)

def row_seperation_loss(features, labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[0,labels[0,:].bool()]
    negatives = similarity_matrix[0,~labels[0,:].bool()]
    for i in range(1, similarity_matrix.shape[0]):
      
      positives = torch.cat((positives, similarity_matrix[i,labels[i,:].bool()]), dim=0)
      negatives = torch.cat([negatives, similarity_matrix[i,~labels[i,:].bool()]], dim=0)

    exp_positives = torch.exp(positives / config.temperature)
    exp_negatives = torch.exp(negatives / config.temperature)
    loss = -torch.log((torch.sum(exp_positives)/(torch.sum(exp_positives)+torch.sum(exp_negatives))))
 
    return loss

def save_codebooks(codebooks):
    np.save('./Codebooks/{0}.npy'.format(args.learned_codebook_name), codebooks.detach().cpu().numpy())

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Using device:", device)
  args = parser.parse_args()
  writer = SummaryWriter()
  # Load config.yml
  with open(os.path.join('./runs/{0}/config.yml'.format(args.folder_name))) as file:
    config = yaml.load(file, Loader=yaml.Loader)

  cp_epoch = '{:04d}'.format(args.pretrain_epochs)
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


  model.cuda()
  print(model)
  # Load weight file
  checkpoint = torch.load('./runs/{0}/checkpoint_{1}.pth.tar'.format(args.folder_name, cp_epoch), map_location=device)
  state_dict = checkpoint['state_dict']
  state_dict_cpy = state_dict.copy()

  # remove prefix
  for k in list(state_dict.keys()):
    if k.startswith('fc.'):
      del state_dict[k]

  log = model.load_state_dict(state_dict, strict=False)
  # assert log.missing_keys == []

  # Load dataset to loader
  if config.dataset_name == 'cifar10':
    codeword_gen_loader, train_loader, val_loader, test_loader = get_cifar10_data_loaders(download=True)
  elif config.dataset_name == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=True)
  elif config.dataset_name == 'mnist':
    codeword_gen_loader, train_loader, val_loader, test_loader = get_mnist_data_loaders(download=True)
  elif config.dataset_name == 'fashion-mnist':
    codeword_gen_loader, train_loader, val_loader, test_loader = get_fashion_mnist_data_loaders(download=True)
  elif config.dataset_name == 'gtsrb':
    codeword_gen_loader, train_loader, val_loader, test_loader = get_gtsrb_data_loaders(download=True)
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
  logging.basicConfig(filename=os.path.join('./runs/{0}'.format(args.folder_name), '({0})acl_cpfc_finetune_training.log'.format(args.loss_type)), level=logging.DEBUG)
  logging.info('({0})acl_cfpc_{1}'.format(args.loss_type, args.folder_name))

  best_val_acc = 0
  test_acc = 0
  best_epoch = 0
  best_model = copy.deepcopy(model.state_dict())
  best_optimizer = copy.deepcopy(optimizer.state_dict())
  for epoch in range(1, args.epochs+1):
    top1_train_accuracy = 0
    selfacc = 0
    codebook = generate_codeword_torch(model, codeword_gen_loader, class_num=class_num, out_dim=code_dim)
    # training
    for counter, (x_batch, y_batch) in enumerate(train_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      logits = model(x_batch)
      
      hl = get_hinge_loss(model(x_batch), codebook, labels=y_batch)
      rsl = row_seperation_loss(logits, y_batch)
      pred_class = ecoc_decodering(logits, codebook)
      ce = criterion(pred_class, y_batch) 
      loss = ce + hl + rsl

      top1 = accuracy(pred_class, y_batch, topk=(1,))

      top1_train_accuracy += top1[0]
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    val_top1_accuracy = 0
    val_top5_accuracy = 0

    codebook_test = generate_codeword_torch(model, codeword_gen_loader, class_num=class_num, out_dim=code_dim)

    # valiadating
    for counter, (x_batch, y_batch) in enumerate(val_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.type(torch.LongTensor).to(device)

      logits = ecoc_decodering(model(x_batch), codebook_test)
      top1, top5 = accuracy(logits, y_batch, topk=(1,5))
      val_top1_accuracy += top1[0]
      val_top5_accuracy += top5[0]

    val_top1_accuracy /= (counter + 1)
    val_top5_accuracy /= (counter + 1)

    # testing
    for counter, (x_batch, y_batch) in enumerate(test_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.type(torch.LongTensor).to(device)

      logits = ecoc_decodering(model(x_batch), codebook_test)
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
    
  checkpoint_name = 'acl_cfpc_best_checkpoint_{0}.pth.tar'.format(best_epoch)
  save_checkpoint({
      'epoch': best_epoch,
      'arch': args.arch,
      'state_dict': best_model,
      'optimizer': best_optimizer,
  }, is_best=False, filename='./runs/{0}/{1}'.format(args.folder_name, checkpoint_name))
  logging.info("best_epoch = {0}, best_val_acc = {1}, test_acc = {2}".format(best_epoch, best_val_acc, test_acc))
  print("({0})acl_cfpc best_epoch = {2}, best_val_acc = {3}, test_acc = {4}".format(args.loss_type, args.folder_name, best_epoch, best_val_acc, test_acc))

  codebook = generate_codeword_torch(model, codeword_gen_loader, class_num=class_num, out_dim=code_dim)
  save_codebooks(codebook)
