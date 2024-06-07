import torch.nn as nn
import torch
import torch.nn.functional as F


#unet的基本结构，
class UpLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out, ndims):

        super(UpLayer, self).__init__()
        conv_op = nn.ConvTranspose2d if ndims == 2 else nn.ConvTranspose3d
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

class ResUNetDecode(nn.Module):
    def __init__(self, config ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.config = config
        num_input_channels = config["num_input_channels"]
        first_layer_channels = config["first_layer_channels"]
        ndims = config["ndims"]
        steps = config["steps"]
        self.output = None

        self.map = nn.Sequential(
            nn.Conv3d(first_layer_channels, first_layer_channels, 3, 1, padding=1),
            nn.Conv3d(first_layer_channels, 1, 3 , 1, padding=1),
            nn.Sigmoid()
        )

        #link_chanels
        decoder_Up = [UpLayer(first_layer_channels * 2, first_layer_channels, ndims)]
        decoder_Conv = [ConvLayer(first_layer_channels *2, first_layer_channels , ndims)]


        for i in range(1, steps):
            up = UpLayer(first_layer_channels * 2 ** (i + 1), first_layer_channels * 2 ** (i), ndims)
            en_conv = ConvLayer(first_layer_channels * 2 ** (i+1),first_layer_channels * 2 ** (i), ndims)
            decoder_Up.append(up)
            decoder_Conv.append(en_conv)

        #输出
        self.decoder_Up = nn.ModuleList(decoder_Up)
        self.decoder_Conv =  nn.ModuleList(decoder_Conv)


    def forward(self, input):
        assert len(input) == len(self.decoder_Up)+1
        x = input[-1]
        link = []

        for i in range(len(self.decoder_Up)-1,-1,-1):
            # resnet
            x = F.dropout(x, self.dorp_rate, self.training)
            x = self.decoder_Up[i](x)
            out = torch.cat([x, input[i]], dim=1)
            x = self.decoder_Conv[i](out)+x
            link.append(x)
        x = self.map(x)

        self.output = link
        return x



if __name__ == "__main__":
    from scripts.utils import load_config
    from ResUnetEncode import ResUNetEncode

    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/mappingNet01.yaml")
    encode = ResUNetEncode(config["feature_extractor"],training=True)
    decode = ResUNetDecode(config["feature_extractor"],training=True)

    x = torch.rand(16, 2, 64, 64, 64).cuda()
    encode = encode.cuda()
    decode = decode.cuda()
    out = encode.forward(x)
    decode.forward(encode.output)
    print(out)

 

