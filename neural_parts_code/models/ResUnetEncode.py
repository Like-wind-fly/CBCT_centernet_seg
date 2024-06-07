import torch.nn as nn
import torch
import torch.nn.functional as F


#unet的基本结构，
class DownLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out, ndims):

        super(DownLayer, self).__init__()
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        conv = conv_op(num_channels_in, num_channels_out, 2, 2)
        RELU = nn.PReLU(num_channels_out)

        self.unet_layer = nn.Sequential(conv,RELU)

    def forward(self, x):
        return self.unet_layer(x)

#unet的基本结构，
class ConvLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out, ndims):

        super(ConvLayer, self).__init__()
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        conv1 = conv_op(num_channels_in, num_channels_out, 3, 1, padding=1)
        conv2 = conv_op(num_channels_out, num_channels_out, 3, 1, padding=1)
        conv3 = conv_op(num_channels_out, num_channels_out, 3, 1, padding=1)
        RELU1 = nn.PReLU(num_channels_out)
        RELU2 = nn.PReLU(num_channels_out)
        RELU3 = nn.PReLU(num_channels_out)

        self.unet_layer = nn.Sequential(conv1,RELU1,conv2,RELU2,conv3,RELU3)

    def forward(self, x):
        return self.unet_layer(x)

class PoolLine(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, ndims):
        super(PoolLine, self).__init__()
        pool = nn.AdaptiveAvgPool2d if ndims == 2 else nn.AdaptiveAvgPool3d
        self.avgpool = pool(1)

        self.fc = nn.Sequential(
            nn.Linear(num_channels_in, num_channels_in),
            nn.ReLU(),
            nn.Linear(num_channels_out, num_channels_out)
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x



class ResUNetEncode(nn.Module):
    def __init__(self, config ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2
        self.feature_size = config["feature_size"]

        self.config = config
        self.output = None

        num_input_channels = config["num_input_channels"]
        first_layer_channels = config["first_layer_channels"]
        ndims = config["ndims"]
        steps = config["steps"]


        #最大池化，用来下采样
        #link_chanels
        encoder_Conv = [ConvLayer(num_input_channels, first_layer_channels, ndims)]
        encoder_Down = [DownLayer(first_layer_channels,first_layer_channels*2,ndims)]
        # link_chanels = [config.first_layer_channels]

        for i in range(1, steps + 1):
            en_conv = ConvLayer(first_layer_channels * 2 ** (i),first_layer_channels * 2 ** (i), ndims)
            down = DownLayer(first_layer_channels * 2 ** (i),first_layer_channels * 2 ** (i+1), ndims)
            encoder_Conv.append(en_conv)
            encoder_Down.append(down)
            # link_chanels.append(config.first_layer_channels * 2 ** (i))

        self.map = PoolLine(first_layer_channels * 2 ** (steps+1),config["feature_size"],ndims)

        #输出
        self.encoder_Conv =  nn.ModuleList(encoder_Conv)
        self.encoder_Down =  nn.ModuleList(encoder_Down)

    def forward(self, x):
        link = []
        for i in range(0,self.config["steps"]+1):
            # resnet
            if i == 0:
                out = self.encoder_Conv[i](x)
            else :
                out = self.encoder_Conv[i](x) + x
            out = F.dropout(out, self.dorp_rate, self.training)
            link.append(out)
            x = self.encoder_Down[i](out)

        self.output = link
        x = self.map(x)
        return x

if __name__ == "__main__":
    from utils import load_config

    config = load_config("../config/mappingNet01.yaml")
    feature_extractor = ResUNetEncode(config["feature_extractor"],training=True)

    x = torch.rand(16, 2, 64, 64, 64).cuda()
    net = feature_extractor.cuda()
    out = net.forward(x)
    print(out)

 

 

