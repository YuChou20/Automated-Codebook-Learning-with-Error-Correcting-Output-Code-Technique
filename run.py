import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from models.resnet_ecoc_simclr import ResNetECOCSimCLR
from simclr import SimCLR
from lars import LARS

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--lr', '--learning-rate', default=1.0, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--log-every-n-steps', default=1, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--model_version', default=4, type=int, help='Model version.'
                    'Version 2: Replace conv 7x7 with conv 3x3, and remove first max pooling.'
                    'Version 3: Add average column spearation loss.'
                    'Version 4: Column separation loss with csw.'
                    'Version 5: Add ECOC encoder layer.')
parser.add_argument('--csw', default=0.001, type=float, help='column seperation loss weight.')
parser.add_argument('--n_neighbors', default=2048, type=int, help='n neighbor for consine similarity search.')
parser.add_argument('--save_weight_every_n_steps', default=50, type=int,
                    help='Save weight every n steps')


def main():
    import warnings
    warnings.filterwarnings("ignore")

    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    if args.model_version == 5:
        model = ResNetECOCSimCLR(base_model=args.arch, out_dim=args.out_dim)
    else:
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer = LARS(
        model.parameters(),
        # lr=0.3*args.batch_size/256,
        lr=args.lr,
        weight_decay=args.weight_decay,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
