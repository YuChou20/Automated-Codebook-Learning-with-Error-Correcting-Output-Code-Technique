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
import faiss
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
parser.add_argument('-folder_name', default='cifar10-baseline-out100',
                    help='model file name')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--pretrain_epochs', default=1800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-loss_type', default='CE+HL',
                    help='model loss type')

# Attack settings
parser.add_argument('--attack_type', default='None',
                    help='FGSM' or 'PGD' or 'CWL2 or None', choices=['FGSM', 'PGD', 'CWL2', 'None'])
parser.add_argument('--norm', default=np.inf , type=int, metavar='N',
                    help='norm of the attack (np.inf or 2)')
parser.add_argument('--max_iter', default=200, type=int, metavar='N',
                    help='max iteration for PGD attack ')
parser.add_argument('--epsilon', default=0.06, type=int, metavar='N',
                    help='bound the attack norm ')
parser.add_argument('--eps_step', default=0.02, type=int, metavar='N',
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
    

def generate_codeword_with_infonce(model, data, class_num=10, out_dim=2048):
  # Record the codeword for each category
  codewords = np.zeros((class_num, out_dim))
  # Record how many samples there are for each category
  num_of_class = np.zeros(class_num)
  positives = torch.Tensor().to(device)
  negatives = torch.Tensor().to(device)
  for x, y in data:
    x = x.to(device)
    # Get outputs of ecoc encoder
    logits = model(x)
    exp_positives, exp_negatives = info_nce_loss_for_codebook(logits, y)
    positives = torch.cat((positives, exp_positives), dim=0)
    negatives = torch.cat((negatives, exp_positives), dim=0)
    # positives += exp_positives
    # negatives += exp_negatives
    for c in range(class_num):
      # Get the sample index belonging to class c
      yi = (y == c).nonzero(as_tuple=True)
      num_of_class[c] += len(yi[0])
      # Sum the current codeword of class c
      codewords[c,:] += torch.sum(logits[yi],dim=0).data.cpu().numpy()
  # Average the codeword
  for i in range(class_num):
    codewords[i] /= num_of_class[i]

  # convert code to -1 and 1
  # codewords[codewords>0]=1
  # codewords[codewords<=0]=-1
  # for i in range(10):
  #   for j in range(i, 10):
  #     c = 0
  #     if i == j:
  #       continue
  #     for k in range(2048):
  #       c += abs(codewords[i,k]-codewords[j,k])
  #     print('class {0}, and class {1} have {2} differents.'.format(i,j,c))
  # return torch.tensor(codewords, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
  infoNCE = -torch.log((torch.sum(positives)/(torch.sum(positives)+torch.sum(negatives))))
  return codewords, infoNCE

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

  # convert code to -1 and 1
  # codewords[codewords>0]=1
  # codewords[codewords<=0]=-1
  # for i in range(10):
  #   for j in range(i, 10):
  #     c = 0
  #     if i == j:
  #       continue
  #     for k in range(2048):
  #       c += abs(codewords[i,k]-codewords[j,k])
  #     print('class {0}, and class {1} have {2} differents.'.format(i,j,c))
  # return torch.tensor(codewords, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
  return codewords

def calculate_cosine_similarity(features, codewords):
    codewords = torch.tensor(codewords, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
    # normalize features and codewords by their L2 norm
    features = features / torch.norm(features, dim=1, keepdim=True)
    codewords = codewords / torch.norm(codewords, dim=1, keepdim=True)
    return torch.einsum('ij,kj->ik', features, codewords)

def get_logits_test(features, codewords, out_dim=10):
    # Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
    #                          = dot(u / norm(u), v / norm(v))
    # We fist normalize the rows, before computing their dot products via transposition:
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = codewords / codewords.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    # print(res[0,:])
    return torch.tensor(res, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
    # logits = nn.functional.cosine_similarity(features, codewords, dim=-1)
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # logits = cos(features, codewords)
    # print(logits.shape)
    # return torch.tensor(logits, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)

def get_logits(features, codewords, out_dim=10):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # consor data type from tensor to np array
    # np_features = features.detach().cpu().numpy().astype('float32')
    np_features = features.data.cpu().numpy().astype('float32')
    # Eliminate warning
    codebook = np.zeros((39, 2048)).astype('float32')
    codebook[:10,:] = codewords.astype('float32')
    codewords = codebook
    # cosine similarity search using faiss tool
    vector_dim = codewords.shape[1]
    faiss.normalize_L2(np_features)
    faiss.normalize_L2(codewords)
    
    nlist = 1
    quantizer = faiss.IndexFlatL2(vector_dim) 
    index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT) 
    index.nprobe = 1
    index.train(codewords) 
    index.add(codewords)
    D, I = index.search(np_features, out_dim)

    ind = I.argsort(axis=-1)
    # Order according to the class(not similarity)
    logits = np.take_along_axis(D, ind, axis=-1)
    return torch.tensor(logits, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)

def get_hinge_loss(features, codewords, labels):
    loss = 0
    for i in range(features.shape[0]):
      cos = torch.nn.CosineSimilarity(dim=0)
      y = torch.tensor(codewords[labels[i]], dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)
      loss += max(0.5-cos(features[i], y),0)
    return torch.tensor(loss, dtype=torch.float32, device=torch.device('cuda:0'),  requires_grad=True)

# def info_nce_loss(features, labels):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 对标签进行独热编码
#     num_classes = labels.max() + 1
#     # print(labels)
#     labels_one_hot = torch.zeros(labels.size(0), num_classes).to(device)
#     labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
#     # 计算相似度矩阵
#     features = F.normalize(features, dim=1)
#     similarity_matrix = torch.matmul(features, features.T)
    
#     # 计算损失
#     mask = torch.eye(labels.size(0)).bool().to(device)
#     similarity_matrix = similarity_matrix[~mask].view(labels.size(0), -1)
#     print(mask)
#     labels_one_hot = labels_one_hot[~mask].view(labels.size(0), -1)
    
    
#     positives = torch.masked_select(similarity_matrix, labels_one_hot.bool()).view(labels.size(0), -1)
#     negatives = torch.masked_select(similarity_matrix, ~labels_one_hot.bool()).view(labels.size(0), -1)
    
#     logits = torch.cat([positives, negatives], dim=1)
#     labels = torch.zeros(logits.size(0)).long().to(device)
    
#     logits = logits / config.temperature
#     return logits, labels

def info_nce_loss_for_codebook(features, labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

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
    # loss = -torch.log((torch.sum(exp_positives)/(torch.sum(exp_positives)+torch.sum(exp_negatives))))
    return exp_positives, exp_negatives

def info_nce_loss(features, labels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # labels = torch.cat([torch.arange(config.batch_size/2) for i in range(config.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * config.batch_size, self.args.n_views * config.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives

    # print(similarity_matrix[labels.bool()].shape)
    positives = similarity_matrix[0,labels[0,:].bool()]
    negatives = similarity_matrix[0,~labels[0,:].bool()]
    for i in range(1, similarity_matrix.shape[0]):
      # mask = torch.ones(similarity_matrix.shape[0], dtype=torch.bool)
      # print('mask', mask.shape)
      # mask[i] = False
      # print(labels.bool()[mask].shape)
      positives = torch.cat((positives, similarity_matrix[i,labels[i,:].bool()]), dim=0)
      negatives = torch.cat([negatives, similarity_matrix[i,~labels[i,:].bool()]], dim=0)
      # print(positives.shape)
      # print(negatives.shape)
    exp_positives = torch.exp(positives / config.temperature)
    exp_negatives = torch.exp(negatives / config.temperature)
    loss = -torch.log((torch.sum(exp_positives)/(torch.sum(exp_positives)+torch.sum(exp_negatives))))
    # positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # print(type(positives))

    # select only the negatives the negatives
    # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # logits = torch.cat([positives, negatives], dim=1)
    # # print(positives.shape, negatives.shape, logits.shape)
    # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # logits = logits / config.temperature
    # criterion = nn.CrossEntropyLoss().cuda(device)
    # loss  = criterion(logits, labels)
    return loss

def get_acc(logits, labels):
  counts = 0
  for i in range(len(logits)):
    pred_class = torch.argmax(logits[i,:])
    if pred_class == labels:
      counts += 1
  return counts/len(logits)

def get_attacked_dataset(attack_type, model, test_data, codewords):
  if attack_type == 'FGSM':
      data_perturbed = AdversarialAttackCleverHans.FGSM(model, config.batch_size, test_data, args.epsilon, args.norm)             
  elif attack_type == 'PGD':
      data_perturbed = AdversarialAttackCleverHans.PGD(model, config.batch_size, test_data, args.epsilon, args.eps_step, args.max_iter, args.norm, args.loss, early_stop=True)
  elif attack_type == 'CWL2':
      data_perturbed = AdversarialAttackCleverHans.CW_L2(model, config.batch_size, test_data, max_iter=args.max_iter, mode="ECOC-SimCLR", codewords=codewords)
  return data_perturbed

def column_seperation_loss(features):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # consor data type from tensor to np array
    np_features = features.astype('float32')
    # anothor way to convert from tensor to numpy
    # np_features = features.data.cpu().numpy().astype('float32') 

    # cosine similarity search using faiss tool
    vector_dim = np_features.shape[1]
    faiss.normalize_L2(np_features)
    
    nlist = 10
    quantizer = faiss.IndexFlatL2(vector_dim) 
    index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT) 
    index.nprobe = 10
    index.train(np_features) 
    index.add(np_features)
    D, I = index.search(np_features, 10)
    print(D)

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Using device:", device)
  args = parser.parse_args()
  writer = SummaryWriter()
  # Load config.yml
  with open(os.path.join('./runs/{0}/config.yml'.format(args.folder_name))) as file:
    config = yaml.load(file, Loader=yaml.Loader)

  # cp_epoch = (4-len(str(config.epochs)))*'0' + str(config.epochs)
  cp_epoch = '{:04d}'.format(args.pretrain_epochs)
  code_dim = config.code_dim
  class_num = 43 if config.dataset_name == 'gtsrb' else 10
  # Get baseline model arch
  if config.arch == 'resnet18':
    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
  elif config.arch == 'resnet50':
    if config.model_version == 5:
      model = ResNetECOCSimCLR(base_model=config.arch, out_dim=class_num, code_dim=code_dim)
      
      # Remove last relu
      model.ecoc_encoder[1]= nn.Identity()
      model.fc = nn.Identity()
      # dim_mlp = model.ecoc_encoder[0].out_features
      # model.fc = nn.Linear(dim_mlp, 10)
    else:
      model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
      dim_mlp = model.fc.in_features
      model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
      model.maxpool = nn.Identity()
    print(model)


  model.cuda()
  # print(model)
  # Load weight file
  checkpoint = torch.load('./runs/{0}/checkpoint_{1}.pth.tar'.format(args.folder_name, cp_epoch), map_location=device)
  state_dict = checkpoint['state_dict']
  state_dict_cpy = state_dict.copy()

  # remove prefix
  if config.model_version == 5:
    for k in list(state_dict.keys()):
      if k.startswith('fc.'):
        del state_dict[k]
  else:
    for k in list(state_dict.keys()):
      if k.startswith('backbone.'):
        if k.startswith('backbone') and not k.startswith('backbone.fc'):
          # remove prefix
          state_dict[k[len("backbone."):]] = state_dict[k]
      del state_dict[k]

  log = model.load_state_dict(state_dict, strict=False)
  # assert log.missing_keys == []

  # Load dataset to loader
  if config.dataset_name == 'cifar10':
    codeword_gen_loader, train_loader, val_loader, test_loader = get_cifar10_data_loaders(download=True)
  elif config.dataset_name == 'stl10':
    train_loader, test_loader = get_stl10_data_loaders(download=True)
  elif config.dataset_name == 'mnist':
    codeword_gen_loader, train_loader, test_loader = get_mnist_data_loaders(download=True)
  elif config.dataset_name == 'fashion-mnist':
    codeword_gen_loader, train_loader, test_loader = get_fashion_mnist_data_loaders(download=True)
  elif config.dataset_name == 'gtsrb':
    codeword_gen_loader, train_loader, test_loader = get_gtsrb_data_loaders(download=True)
  print("Dataset:", config.dataset_name)
  

  requires_grad_list = ['ecoc_encoder.0.weight', 'ecoc_encoder.0.bias'] if config.model_version == 5 else []

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
  # optimizer = torch.optim.Adam(model.parameters(), lr=5, weight_decay=0.0008)
  criterion = torch.nn.CrossEntropyLoss().to(device)

  # training & testing
  logging.basicConfig(filename=os.path.join('./runs/{0}'.format(args.folder_name), '(val)ecocdec_eval_{0}_{1}.log'.format(args.attack_type,cp_epoch)), level=logging.DEBUG)
  logging.info('({0}){1}'.format(args.loss_type, args.folder_name))
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
  print(len(val_loader))
  # Show cosine similarity between codeword.
  # column_seperation_loss(codewords)
  for epoch in range(1, args.epochs+1):
    # if epoch % 20 == 0:
    
    top1_train_accuracy = 0
    selfacc = 0
    
    # training
    for counter, (x_batch, y_batch) in enumerate(train_loader):
      # codewords, infoNCE = generate_codeword_with_infonce(model, codeword_gen_loader, class_num=10, out_dim=code_dim)
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      logits = model(x_batch)
      infoNCE = info_nce_loss(logits, y_batch)
      pred_class = calculate_cosine_similarity(logits, codewords)
      # loss = criterion(pred_class, y_batch)+get_hinge_loss(model(x_batch), codewords, labels=y_batch)+infoNCE
      loss = get_hinge_loss(model(x_batch), codewords, labels=y_batch) + infoNCE
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
    attacked_top1_accuracy = 0
    attacked_top5_accuracy = 0

    codewords_test = generate_codeword(model, codeword_gen_loader, class_num=class_num, out_dim=code_dim)
    best_val_acc = 0
    test_acc = 0
    best_epoch = 0
    best_model = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    # valiadating
    for counter, (x_batch, y_batch) in enumerate(val_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.type(torch.LongTensor).to(device)

      # logits = get_logits(model(x_batch), codewords)
      logits = calculate_cosine_similarity(model(x_batch), codewords_test)

      top1, top5 = accuracy(logits, y_batch, topk=(1,5))
      val_top1_accuracy += top1[0]
      val_top5_accuracy += top5[0]

    val_top1_accuracy /= (counter + 1)
    val_top5_accuracy /= (counter + 1)

    # testing
    for counter, (x_batch, y_batch) in enumerate(test_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.type(torch.LongTensor).to(device)

      # logits = get_logits(model(x_batch), codewords)
      logits = calculate_cosine_similarity(model(x_batch), codewords_test)

      top1, top5 = accuracy(logits, y_batch, topk=(1,5))
      top1_accuracy += top1[0]
      top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)
    if val_top1_accuracy > best_val_acc:
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
    # Get attacked dataset if attack
    if args.attack_type != 'None':
      perturbed_test_loader = get_attacked_dataset(attack_type=args.attack_type, model=model, test_data=test_loader, codewords=codewords_test)
      for counter, (x_batch, y_batch) in enumerate(perturbed_test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.type(torch.LongTensor).to(device)

        # logits = get_logits(model(x_batch), codewords)
        logits = calculate_cosine_similarity(model(x_batch), codewords_test)

        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
        attacked_top1_accuracy += top1[0]
        attacked_top5_accuracy += top5[0]

      attacked_top1_accuracy /= (counter + 1)
      attacked_top5_accuracy /= (counter + 1)

      writer.add_scalar('{0}_attack_loss'.format(args.attack_type), loss, global_step=epoch)
      writer.add_scalar('{0}_attack_Eval Train: acc/top1'.format(args.attack_type), top1_train_accuracy, global_step=epoch)
      writer.add_scalar('{0}_attack_Eval: acc/top1'.format(args.attack_type), attacked_top1_accuracy, global_step=epoch)
      writer.add_scalar('{0}_attack_Eval: acc/top5'.format(args.attack_type), attacked_top5_accuracy, global_step=epoch)
      logging.info(f"\t\t\t\t\tAttacked Top1 Test accuracy: {attacked_top1_accuracy.item()}\tTop5 test acc: {attacked_top5_accuracy.item()}")
      print(f"\t\t\t\t\tAttacked Top1 Test accuracy: {attacked_top1_accuracy.item()}\tTop5 test acc: {attacked_top5_accuracy.item()}")

  checkpoint_name = '({0}){1}_best_checkpoint_{2}.pth.tar'.format(args.loss_type, args.folder_name, best_epoch)
  save_checkpoint({
      'epoch': best_epoch,
      'arch': args.arch,
      'state_dict': best_model,
      'optimizer': best_optimizer,
  }, is_best=False, filename='./runs/{0}/{1}'.format(args.folder_name, checkpoint_name))
  logging.info("best_epoch = {0}, best_val_acc = {1}, test_acc = {2}".format(best_epoch, best_val_acc, test_acc))
  print("({0}){1} best_epoch = {2}, best_val_acc = {3}, test_acc = {4}".format(args.loss_type, args.folder_name, best_epoch, best_val_acc, test_acc))