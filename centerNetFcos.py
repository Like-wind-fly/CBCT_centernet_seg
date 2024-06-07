import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_parts_code.models.Encode import Encode
from neural_parts_code.models.Decode import Decode
from neural_parts_code.models.common_models import ConvLayer
from neural_parts_code.stats_logger import StatsLogger

class CenterNetFcosBuilder(object):
    def __init__(self, config):
        self.config = config
        self._encode  = None
        self._decode  = None
        self._town    = None
        self._head    = None
        self._network = None
        self.training = True

    @property
    def encode(self):
        if self._encode is None:
            self._encode = Encode(self.config["Unet"],training= self.training)
        return self._encode

    @property
    def decode(self):
        if self._decode is None:
            self._decode = Decode(self.config["Unet"],training= self.training)
        return self._decode

    @property
    def town(self):
        if self._town is None:
            blockNum = self.config["town"].get("blockNum", 3)
            num_channels_in = self.config["Unet"]["first_layer_channels"] * 2 ** self.config["town"]["used_layer"]
            num_channels_out = self.config["town"]["features"]
            self._town = ConvLayer(num_channels_in, num_channels_out, 3, blockNum)
        return self._town

    @property
    def head(self):
        if self._head is None:
            num_channels_in = self.config["town"]["features"]
            classNum = self.config["data"]["classNum"]
            labelNum = self.config["data"]["labelNum"]

            heatmap = nn.Sequential(
                nn.Conv3d(num_channels_in, num_channels_in, 3, 1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv3d(num_channels_in, num_channels_in, 3, 1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv3d(num_channels_in, classNum, 1, 1),
                nn.Sigmoid()
            )

            regress_town = nn.Sequential(
                nn.Conv3d(num_channels_in, num_channels_in, 3, 1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv3d(num_channels_in, num_channels_in, 3, 1, padding=1),
                nn.LeakyReLU(0.2),
            )

            classify = nn.Sequential(
                nn.Conv3d(num_channels_in, labelNum , 1, 1),
                nn.Sigmoid()
            )

            reg = nn.Sequential(
                nn.Conv3d(num_channels_in, 3, 1, 1)
            )
            dwh = nn.Sequential(
                nn.Conv3d(num_channels_in, 3, 1, 1)
            )



            self._head = [heatmap,reg,dwh,classify,regress_town]

        return self._head

    @property
    def network(self):
        if self._network is None:
            self._network = CenterNet(
                self.encode,
                self.decode,
                self.town,
                self.head,
                self.config
            )
        return self._network

class CenterNet(nn.Module):
    def __init__(self, encode,decode,town,head,config):
        super(CenterNet, self).__init__()
        self.encoder = encode
        self.decoder = decode
        self.town = town
        heatmap,reg,dwh,classify,regress_town = head
        self.heatmap = heatmap
        self.reg = reg
        self.dwh = dwh
        self.classify = classify
        self.regress_town = regress_town
        self.config = config

    def forward(self, input):
        # read in data                
        if input.ndim == 4: input = input.unsqueeze(1)
        assert input.ndim == 5

        # propagate UNET
        self.encoder(input)
        link = self.encoder.output
        self.decoder(link)
        features = self.decoder.output

        layer_num = len(features)
        used_layer = self.config["town"]["used_layer"]
        feature_map = features[layer_num - used_layer -1]
        feature_map = self.town(feature_map)

        hm = self.heatmap(feature_map)

        feature_map = self.regress_town(feature_map)
        reg = self.reg(feature_map) / 2**used_layer
        dwh = self.dwh(feature_map)
        classify = self.classify(feature_map)

        predictions = {}
        predictions["heatmap"] = hm
        predictions["reg"] = reg
        predictions["dwh"] = dwh

        predictions["classify"] = classify

        return predictions



def train_on_batch(network, optimizer, loss_fn, metrics_fn, X, targets,
                   config):
    """Perform a forward and backward pass on a batch of samples and compute
    the loss.
    """
    optimizer.zero_grad()

    # Extract the per primitive features
    predictions  = network(X)

    # Do the forward pass to predict the primitive_parameters
    batch_loss = loss_fn(predictions, targets, config["loss"])
    metrics_fn(predictions, targets)
    # Do the backpropagation
    batch_loss.backward()

    nn.utils.clip_grad_norm_(network.parameters(), 1)
    # Do the update
    optimizer.step()

    StatsLogger.instance()["optimizer"].value = optimizer.optimizer.param_groups[0]["lr"]*10000
    return batch_loss.item()


def validate_on_batch(network, loss_fn, metrics_fn, X, targets, config):
    # Extract the per primitive features
    predictions  = network(X)

    # Do the forward pass to predict the primitive_parameters
    batch_loss = loss_fn(predictions, targets, config["loss"])
    metrics_fn(predictions, targets)

    return batch_loss.item()

if __name__ == "__main__":
    from scripts.utils import load_config
    from thop import profile
    from thop import clever_format

    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/centernet_Unet_layer3.yaml")
    builder = CenterNetBuilder(config)
    net = builder.network.cuda()

    x = torch.rand(1, 1, 160, 256, 256).cuda()

    macs, params = profile(net, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    # prediction = net(x)

    print(macs)
    print(params)
