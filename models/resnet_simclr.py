import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError
from collections import OrderedDict

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.backbone.maxpool = nn.Identity()
        # # add ecoc encoder
        # self.backbone.ecoc_encoder = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU())
        # # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.backbone.fc)

        # version 5 & 6
        # self.backbone.fc = nn.Identity(nn.Linear(dim_mlp, dim_mlp))
        # self.ecoc_encoder = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True))
        # self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), nn.Linear(dim_mlp, 128))
        # print(self.backbone)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
        # x = self.backbone(x)
        # x = self.ecoc_encoder(x)
        # x = self.fc(x)
        # return x
