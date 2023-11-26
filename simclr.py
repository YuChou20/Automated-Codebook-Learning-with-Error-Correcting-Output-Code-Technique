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
        self.model = kwargs['model'].to(self.args.device)

        # self.model1 = copy.deepcopy(self.model)
        # # load weight if need
        # self.load_model_weight('./runs/cifar10-1000-lars-v2-2/checkpoint_1000.pth.tar')
        # self.model2 = copy.deepcopy(self.model)
        # self.compare_models(self.model1, self.model2)

        self.model.backbone.avgpool.register_forward_hook(self.hook)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.csw = kwargs['csw']
        self.model_version =  kwargs['model_version']
        # The simensions of hidden layer activation only 2048 
        self.n_neighbors = kwargs['n_neighbors']+1 if kwargs['n_neighbors'] < 2048 else 2048
        self.activation = torch.empty([1, 1]) 
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')
        else:
            print('Models different!')

    def load_model_weight(self, path):
        print('load weight file from {0}'.format(path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=False)

    def hook(self, module, input, output):
        self.activation = output.detach()
        self.activation = self.activation.view([512,2048])

    def column_seperation_loss(self, features):
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        # consor data type from tensor to np array
        np_features = features.detach().cpu().numpy().astype('float32')
        # the feature need to transpose and convert data distribution to continuous
        np_features = np.ascontiguousarray(np_features.T)
        # cosine similarity search using faiss tool
        vector_dim = np_features.shape[1]
        faiss.normalize_L2(np_features)
        
        nlist = 100
        quantizer = faiss.IndexFlatL2(vector_dim) 
        index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT) 
        index.nprobe = 100
        index.train(np_features) 
        index.add(np_features)
        D, I = index.search(np_features, self.n_neighbors)
        if self.model_version==3:
            loss = D[:,1:].sum()/ (np_features.shape[0]*self.n_neighbors)
        elif self.model_version==4:
            loss = D[:,1:].sum()*self.csw
        print('csl: ', loss)
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

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        # print(self.model)
        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    csl = self.column_seperation_loss(self.activation)
                    logits, labels = self.info_nce_loss(features)
                    infoNCE = self.criterion(logits, labels)
                    loss = infoNCE.add(csl)
                    # loss = infoNCE

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
