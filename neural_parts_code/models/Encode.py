import torch.nn as nn
import torch
import torch.nn.functional as F
from neural_parts_code.models.common_models import ConvLayer,DownLayer



class Encode(nn.Module):
    def __init__(self, config ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.config = config
        self.output = None

        num_input_channels = config["num_input_channels"]
        first_layer_channels = config["first_layer_channels"]
        ndims = config["ndims"]
        steps = config["steps"]
        blockNum = config.get("blockNum", 3)
        lu = config.get("lu", "LeakyReLU")


        #最大池化，用来下采样
        #link_chanels
        encoder_Conv = [ConvLayer(num_input_channels, first_layer_channels, ndims,blockNum,lu = lu)]
        encoder_Down = [DownLayer(first_layer_channels,first_layer_channels*2,ndims,lu = lu)]
        # link_chanels = [config.first_layer_channels]

        for i in range(1, steps + 1):
            en_conv = ConvLayer(first_layer_channels * 2 ** (i),first_layer_channels * 2 ** (i), ndims,blockNum,lu = lu)
            down = DownLayer(first_layer_channels * 2 ** (i),first_layer_channels * 2 ** (i+1), ndims,lu = lu)
            encoder_Conv.append(en_conv)
            encoder_Down.append(down)
            # link_chanels.append(config.first_layer_channels * 2 ** (i))


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

        return x

if __name__ == "__main__":
    from scripts.utils import load_config

    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/corticalflow.yaml")
    encode = Encode(config["deform0"],training=True).cuda()

    x = torch.rand(16, 1, 64, 64, 64).cuda()
    net = encode.cuda()
    out = net.forward(x)
    print(out)

 

 

