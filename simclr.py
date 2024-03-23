import logging
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from torchvision.transforms import Resize
import faiss
import copy

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model_version =  self.args.model_version#kwargs['model_version']
        self.model = kwargs['model'].to(self.args.device)
        if self.args.load_weight:
            checkpoint = torch.load(self.args.weight_path, map_location=self.args.device)
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict, strict=False)

        if self.model_version == 5:
            self.model.ecoc_encoder[0].register_forward_hook(self.hook)
        else:
            self.model.backbone.avgpool.register_forward_hook(self.hook)

        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.csw = self.args.csw
        self.code_dim = self.args.code_dim
        
        # The simensions of hidden layer activation only 2048
        # self.n_neighbors = self.args.n_neighbors+1 if self.args.n_neighbors < 2048 else 2048
        self.n_neighbors = self.code_dim
        self.weight_save_epoch = self.args.save_weight_every_n_steps
        self.activation = torch.empty([1, 1]) 
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def hook(self, module, input, output):
        self.activation = output.detach()
        self.activation = self.activation.view([self.args.batch_size*2,self.code_dim])

    def column_seperation_loss_gpu(self, feature):
        features_T = torch.transpose(feature.detach(), 0, 1)
        print('T shape', features_T.shape)
        norm_features = features_T / torch.norm(features_T, dim=1, keepdim=True)
        print('zero0', torch.is_nonzero(features_T))
        print('check:', torch.isnan(norm_features).any())
        cosine_similarity_matrix = torch.einsum('ij,kj->ik', norm_features, norm_features)

        for i in range(2048):
            cosine_similarity_matrix[i,i] = 0.0

        print(torch.isnan(cosine_similarity_matrix).any())
        loss = cosine_similarity_matrix.flatten().sum(0)
        # loss = torch.sum(torch.sum(cosine_similarity_matrix, 1),0).type(torch.float32).to("cuda:0")

        # torch.sum(cosine_similarity_matrix).type(torch.float32).to("cuda:0")
        #cosine_similarity_matrix.sum(1).sum(0).type(torch.float32).to("cuda:0")
        print( torch.any(cosine_similarity_matrix.isnan()))
        # print(cosine_similarity_matrix.data.cpu().numpy().astype('float32').sum())
        # print('loss:', loss)


        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        n = cos(features_T[0,:], features_T[1,:])
        # print('proof: ', n)
        return loss

    def column_seperation_loss(self, features):
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        # consor data type from tensor to np array
        np_features = features.detach().cpu().numpy().astype('float32')
        # anothor way to convert from tensor to numpy
        # np_features = features.data.cpu().numpy().astype('float32') 

        # the feature need to transpose and convert data distribution to continuous
        np_features = np.ascontiguousarray(np_features.T)
        # cosine similarity search using faiss tool
        vector_dim = np_features.shape[1]
        faiss.normalize_L2(np_features)
        
        nlist = self.code_dim if self.code_dim <= 100 else 100
        quantizer = faiss.IndexFlatL2(vector_dim) 
        index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT) 
        index.nprobe = self.code_dim if self.code_dim <= 100 else 100
        index.train(np_features) 
        index.add(np_features)
        D, I = index.search(np_features, self.n_neighbors)

        if self.model_version==3:
            loss = D[:,1:].sum()/ (np_features.shape[0]*self.n_neighbors)
        elif self.model_version==4 or 5:
            loss = D[:,1:].sum()*self.csw
        return torch.tensor(loss, dtype=torch.float32, device=torch.device('cuda:0'))
        


    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    def train(self, train_loader):
        # print(self.model)
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        
        for epoch_counter in range(1, self.args.epochs+1):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    infoNCE = self.criterion(logits, labels)
                    if self.model_version == 2:
                        loss = infoNCE
                    else:
                        # calculate csl
                        csl = self.column_seperation_loss(self.activation)
                        # print('cpu: ', csl)
                        # csl = self.column_seperation_loss_gpu(self.activation)
                        # print('gpu: ', csl)
                        loss = infoNCE.add(csl)

                        # Use for baseline model
                        # loss = infoNCE

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

            if epoch_counter % self.args.log_every_n_steps == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                self.writer.add_scalar('loss', loss, global_step=epoch_counter)
                self.writer.add_scalar('acc/top1', top1[0], global_step=epoch_counter)
                self.writer.add_scalar('acc/top5', top5[0], global_step=epoch_counter)
                self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=epoch_counter)
                
            # warmup for the first 10 epochs
            if epoch_counter > 10:
                self.scheduler.step()
            # save model checkpoints
            if epoch_counter % self.weight_save_epoch == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                print('save weight {0}: '.format(checkpoint_name), checkpoint_name)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
