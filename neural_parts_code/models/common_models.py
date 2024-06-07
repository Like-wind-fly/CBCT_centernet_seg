import torch.nn as nn

#unet的基本结构，
class UpLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out, ndims,lu = "LeakyReLU"):

        super(UpLayer, self).__init__()
        conv_op = nn.ConvTranspose2d if ndims == 2 else nn.ConvTranspose3d
        conv = conv_op(num_channels_in, num_channels_out, 2, 2)
        if lu == "LeakyReLU":
            RELU = nn.LeakyReLU(0.2)
        else:
            RELU = nn.PReLU(num_channels_out)

        norm = nn.BatchNorm3d(num_channels_out)
        self.unet_layer = nn.Sequential(conv,norm,RELU)

    def forward(self, x):
        return self.unet_layer(x)

#unet的基本结构，
class DownLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out, ndims,lu = "LeakyReLU"):

        super(DownLayer, self).__init__()
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        conv = conv_op(num_channels_in, num_channels_out, 2, 2)

        if lu == "LeakyReLU":
            RELU = nn.LeakyReLU(0.2)
        else:
            RELU = nn.PReLU(num_channels_out)

        norm = nn.BatchNorm3d(num_channels_out)
        self.unet_layer = nn.Sequential(conv,norm,RELU)

    def forward(self, x):
        return self.unet_layer(x)

#unet的基本结构，
class ConvLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out, ndims, blockNum=3,lu = "LeakyReLU"):

        super(ConvLayer, self).__init__()
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d

        block_list = []
        block_list.append(conv_op(num_channels_in, num_channels_out, 3, 1, padding=1))


        block_list.append(nn.BatchNorm3d(num_channels_out))
        if lu == "LeakyReLU":
            block_list.append(nn.LeakyReLU(0.2))
        else:
            block_list.append(nn.PReLU(num_channels_out))

        for i in range(1,blockNum):
            block_list.append(conv_op(num_channels_out, num_channels_out, 3, 1, padding=1))
            block_list.append(nn.BatchNorm3d(num_channels_out))
            if lu == "LeakyReLU":
                block_list.append(nn.LeakyReLU(0.2))
            else:
                block_list.append(nn.PReLU(num_channels_out))

        self.unet_layer = nn.Sequential(*block_list)

    def forward(self, x):
        return self.unet_layer(x)

if __name__ == "__main__":
    conv = ConvLayer(4,8, 8, 2)
    print("ddd")
