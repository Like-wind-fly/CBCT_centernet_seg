import torch.nn as nn
import torch
import torch.nn.functional as F
from neural_parts_code.models.common_models import ConvLayer,UpLayer



class Decode(nn.Module):
    def __init__(self, config ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.config = config

        first_layer_channels = config["first_layer_channels"]
        ndims = config["ndims"]
        steps = config["steps"]
        blockNum = config.get("blockNum", 3)
        lu = config.get("lu", "LeakyReLU")
        self.output = None

        #link_chanels
        decoder_Up = [UpLayer(first_layer_channels * 2, first_layer_channels, ndims,lu = lu)]
        decoder_Conv = [ConvLayer(first_layer_channels *2, first_layer_channels , ndims, blockNum,lu = lu)]


        for i in range(1, steps):
            up = UpLayer(first_layer_channels * 2 ** (i + 1), first_layer_channels * 2 ** (i), ndims,lu = lu)
            en_conv = ConvLayer(first_layer_channels * 2 ** (i+1),first_layer_channels * 2 ** (i), ndims, blockNum,lu = lu)
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

        self.output = link
        return x



if __name__ == "__main__":
    from scripts.utils import load_config
    from Encode import Encode

    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/corticalflow.yaml")
    encode = Encode(config["deform0"],training=True).cuda()
    decode = Decode(config["deform0"],training=True).cuda()

    x = torch.rand(16, 1, 64, 64, 64).cuda()
    encode = encode.cuda()
    decode = decode.cuda()
    out = encode.forward(x)
    y = decode.forward(out)
    print(out)

 

