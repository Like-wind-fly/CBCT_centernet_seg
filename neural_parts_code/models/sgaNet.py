import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from neural_parts_code.datasets.cbct.common import save_img


from neural_parts_code.models.Encode import Encode
from neural_parts_code.models.Decode import Decode
from neural_parts_code.models.common_models import ConvLayer
from neural_parts_code.stats_logger import StatsLogger
from neural_parts_code.PostProcess.SGA_visualize import get_volume,get_volume2
from neural_parts_code.models.boxDecode import roiCrop
from skimage import filters,measure,morphology
import nibabel as nib
import math



class UNetBuilder(object):
    def __init__(self, config):
        self.config = config
        self._encode  = None
        self._decode  = None
        self._town    = None
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
            num_channels_in = self.config["Unet"]["first_layer_channels"]
            num_channels_out = self.config["town"]["features"]

            self._town = nn.Sequential(
                nn.Conv3d(num_channels_in, num_channels_in, 3, 1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv3d(num_channels_in, num_channels_out, 1, 1)
            )
        return self._town

    @property
    def network(self):
        if self._network is None:
            self._network = UNet(
                self.encode,
                self.decode,
                self.town,
                self.config
            )
        return self._network

class UNet(nn.Module):
    def __init__(self, encode,decode,town,config):
        super(UNet, self).__init__()
        self.encoder = encode
        self.decoder = decode
        self.town = town
        self.config = config

    def forward(self, input):
        # read in data                
        if input.ndim == 4: input = input.unsqueeze(1)
        assert input.ndim == 5

        # propagate UNET
        self.encoder(input)
        link = self.encoder.output
        feature_map = self.decoder(link)
        feature_map = self.town(feature_map)
        feature_map = nn.functional.sigmoid(feature_map)

        predictions = {}
        predictions["quad_seg"] = feature_map
        return predictions


class AttentionUNetBuilder(object):
    def __init__(self, config):
        self.config = config
        self._encode  = None
        self._decode  = None
        self._attention = None
        self._town    = None
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
            num_channels_in = self.config["Unet"]["first_layer_channels"]
            num_channels_out = self.config["town"]["features"]

            self._town = nn.Sequential(
                nn.Conv3d(num_channels_in, num_channels_in, 3, 1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv3d(num_channels_in, num_channels_out, 1, 1)
            )
        return self._town

    @property
    def attention(self):
        if self._attention is None:
            self._attention = Attention_module(self.config)
        return self._attention

    @property
    def network(self):
        if self._network is None:
            self._network = AttentionUNet(
                self.encode,
                self.decode,
                self.town,
                self.attention,
                self.config
            )
        return self._network

class GBA_module(nn.Module):
    def __init__(self , M, C = 8 , training=True):
        super().__init__()
        self.training = training
        self.dorp_rate = 0.2
        self.M = M
        self.C = C

        self.hidden_conv = ConvLayer(M,M, ndims= 3 , blockNum= 1 ,lu = "LeakyReLU")
        self.class_conv = ConvLayer(M, C, ndims=3, blockNum=1, lu="LeakyReLU")
        self.town = nn.Conv1d(C, C, 1, 1)

        self.register_parameter(
            "learnable_weight",
            torch.nn.Parameter(
                torch.randn(2, M, M)
            )
        )

    def forward(self, input):
        B, M , W, H, D = input.shape
        C = self.C

        adjacency_matrix = torch.zeros(C,C).cuda()
        for i in range(0,C - 1):
            adjacency_matrix[i,i]    = 1
            adjacency_matrix[i, i+1] = 1
            adjacency_matrix[i+1, i] = 1

        adjacency_matrix[C - 1, C - 1] = 1

        x_hidden = self.hidden_conv(input)
        x_class  = self.class_conv(input)

        x_hidden = x_hidden.view(B,M,W*H*D).transpose(1,2)

        x_class =  x_class.view(B, C, W * H * D)

        feature = torch.matmul(x_class,x_hidden)

        for i in range(2):
            gcn = torch.matmul(adjacency_matrix, feature)
            gcn = torch.matmul(gcn , self.learnable_weight[i])
            feature = torch.matmul(adjacency_matrix, gcn)

        feature = feature.transpose(1,2)

        x_class = self.town(x_class)
        # x_class = nn.functional.softmax(x_class)
        x_class = nn.functional.sigmoid(x_class)

        x_attention = torch.matmul(feature,x_class)
        x_attention = x_attention.view(B,M,W,H,D)
        x_class = x_class.view(B, C, W, H, D)

        return x_attention,x_class

class GCA_module(nn.Module):
    def __init__(self , M, training=True):
        super().__init__()
        self.training = training
        self.dorp_rate = 0.2
        self.M = M
        self.conv1 = nn.Conv1d(M, M, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(M, M, kernel_size=1, stride=1)


    def forward(self, input):
        feature = input.mean(dim  = [2,3,4]).unsqueeze(dim = 2)
        feature = self.conv1(feature)
        feature = self.conv2(feature)

        feature = feature[:,:,None,None]

        global_context = nn.functional.softmax(feature * input)
        output = input + global_context

        return output


class Attention_module(nn.Module):
    def __init__(self , config):
        super(Attention_module, self).__init__()
        first_layer_channels = config["Unet"]["first_layer_channels"]
        steps = config["Unet"]["steps"]+1
        feature_channels = [first_layer_channels * (2**i) for i in range(steps)]
        self.steps = steps
        GCAs = [GCA_module(feature_channels[i]) for i in range(steps - 1)]
        self.GCAs = nn.ModuleList(GCAs)
        self.GBA = GBA_module(feature_channels[steps - 1 ])

    def forward(self, input):
        assert len(input) == self.steps

        output = []
        for i in range(self.steps - 1):
            feature = self.GCAs[i].forward(input[i])
            output.append(feature)

        feature,classify = self.GBA.forward(input[self.steps - 1])
        output.append(feature)
        return output,classify



class AttentionUNet(nn.Module):
    def __init__(self, encode,decode,town,attention,config):
        super(AttentionUNet, self).__init__()
        self.encoder = encode
        self.decoder = decode
        self.attention = attention
        self.town = town
        self.config = config

    def forward(self, input):
        # read in data
        if input.ndim == 4: input = input.unsqueeze(1)
        assert input.ndim == 5

        B,C,W,H,D = input.shape
        input = input.view(B*C,W,H,D).unsqueeze(dim= 1)

        # propagate UNET
        self.encoder(input)
        link = self.encoder.output
        link,classify = self.attention(link)
        feature_map = self.decoder(link)
        feature_map = self.town(feature_map)
        feature_map = nn.functional.sigmoid(feature_map)

        predictions = {}
        predictions["quad_identify"] = feature_map
        predictions["down_sample_classify"] = classify
        return predictions
# old = model8
# new = model4
# old.quad_seg=new.quad_seg

class SGAnet(nn.Module):
    def __init__(self, quad_seg,guad_identify,config):
        super(SGAnet, self).__init__()
        self.quad_seg = quad_seg
        self.quad_identify = guad_identify
        self.config = config

    def gaussian_kernel(self,size, sigma):
        """生成3D高斯核."""
        # 生成一维的高斯分布
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32,device=torch.device('cuda'))
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        # 为3D生成外积
        g = g[:, None, None] * g[None, :, None] * g[None, None, :]
        # 归一化
        g /= g.sum()
        return g

    def gaussian_blur(self,image, kernel_size=7, sigma=1.0):
        """对3D图像应用高斯模糊."""
        kernel = self.gaussian_kernel(kernel_size, sigma)
        # 高斯核需要是(C_in, C_out, D, H, W)形状
        kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size, kernel_size)
        # padding以保持尺寸不变
        padding = kernel_size // 2
        return F.conv3d(image, kernel, padding=padding, groups=image.shape[1])


    def get_id_ct2(self, ct,seg_4):
        # seg_4[seg_4>0.5] = 1
        seg_4 = torch.round(seg_4)
        seg_4 = self.gaussian_blur(seg_4)
        seg_4 = torch.round(seg_4)
        one_map = torch.ones(seg_4.shape,dtype=seg_4.dtype)  # Assuming seg_4 has shape [batch_size, H, W, D]
        for i in range(4):
            print(i)
            one_map[:,i] = seg_4[:,i,:,:,:]*(i+1)
        one_map=torch.squeeze(one_map,dim=0)
        #最大联通区域去噪
        one_map_list = []
        for i in range(4):
            label = measure.label(one_map[i])
            regions = measure.regionprops(label)
            largest_region = max(regions, key=lambda r: r.area)
            largest_region_voxels = label == largest_region.label
            one_map_list.append(one_map[i]*largest_region_voxels)
        one_map = torch.stack(one_map_list,dim=0)
        print("onemapshape:",one_map.shape)
        save_img(one_map[0].int().cpu().numpy(),
                 "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/sga_net/onemap_001{}.nii".format(1))
        save_img(one_map[1].int().cpu().numpy(),
                 "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/sga_net/onemap_001{}.nii".format(2))
        save_img(one_map[2].int().cpu().numpy(),
                 "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/sga_net/onemap_001{}.nii".format(3))
        save_img(one_map[3].int().cpu().numpy(),
                 "/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/sga_net/onemap_001{}.nii".format(4))

#高斯去噪

        ETMaskDW = one_map.sum(dim=-1) > 0  # Sum over depth
        ETMaskDH = one_map.sum(dim=2) > 0  # Sum over height
        print(one_map.shape)
        rangeDW = masks_to_boxes(ETMaskDW)  # Dummy function to compute bounding boxes
        rangeDH = masks_to_boxes(ETMaskDH)

        rangeDWH = torch.cat([rangeDW[:, [1, 3]], rangeDW[:, [0, 2]], rangeDH[:, [0, 2]]], dim=1)

        dwh = torch.stack([(rangeDWH[:, i * 2 + 1] - rangeDWH[:, i * 2]) for i in range(3)], dim=1)
        dwh[:, 0], dwh[:, 1], dwh[:, 2] = 96, 256, 128  # Preset dimensions for depth, width, and height

        index = torch.stack([(rangeDWH[:, i * 2 + 1] + rangeDWH[:, i * 2]) // 2 for i in range(3)], dim=1).int()

        seg_4 = torch.squeeze(seg_4,dim=0)
        ct = torch.squeeze(ct,dim=0)
        tensor_shape = seg_4.shape[1:]  # Assume seg_4 shape includes batch dimension as first
        out_list = []
        for i in range(4):
            center = index[i]
            sizes = dwh[i]
            mask = one_map[:, i]  # Select the i-th mask for the i-th quadrant
            crop_no_range = [(center[dim] - sizes[dim] // 2).int() for dim in range(3)]
            crop_end_no_range = [(center[dim] + sizes[dim]//2).int() for dim in range(3)]
            crop = [torch.clamp(center[dim] - sizes[dim] // 2, 0, tensor_shape[dim]).int() for dim in range(3)]
            crop_end = [torch.clamp(center[dim] + sizes[dim] // 2, 0, tensor_shape[dim]).int() for dim in range(3)]
            pre = ct[:, crop[0]:crop_end[0], crop[1]:crop_end[1], crop[2]:crop_end[2]]

            padding = []
            for i in range(3):
                left = 0
                right = 0

                if crop_no_range[i] < 0:
                    left -= crop_no_range[i]
                if crop_end_no_range[i] > tensor_shape[i]:
                    right = crop_end_no_range[i] - tensor_shape[i]

                padding.append(right)
                padding.append(left)

            padding.reverse()
            print(padding)
            pad = torch.nn.ConstantPad3d(padding, value=0)

            out = pad(pre.unsqueeze(0).unsqueeze(0))[0, 0]
            print(out.shape)
            out_list.append(out)

        return torch.stack(out_list, dim=0),index # Stack outputs to match the number of quadrants

    def forwardForTraining(self,input):
        quad_ct = input["quad_ct"]
        # identify_ct = input["identify_ct"]

        seg_predcition = {}
        identify_predcition = {}

        seg_predcition = self.quad_seg(quad_ct)
        # identify_predcition = self.quad_identify(identify_ct)
        prediction = dict(list(seg_predcition.items()) + list(identify_predcition.items()))
        return prediction

    def forwardForTesting(self,input):
        quad_ct = input["quad_ct"]
        # identify_ct = input["identify_ct"]
        print(quad_ct.shape)
        seg_predcition = {}
        identify_predcition = {}
        seg_predcition = self.quad_seg(quad_ct)

        resized_tensor,index = self.get_id_ct2(quad_ct,seg_predcition['quad_seg'])

        # new_d, new_h, new_w = 128, 128, 128  #input["identify_ct"].shape
        # resized_tensor = F.interpolate(temp, size=(new_d, new_h, new_w), mode='trilinear', align_corners=False)
        # print("resized_tensor",resized_tensor.shape)
        identify_predcition = self.quad_identify(resized_tensor) #(identify_ct)

        #print(seg_predcition.shape)
        #identify_predcition = self.quad_identify(identify_ct)
        #######
        prediction = dict(list(seg_predcition.items()) + list(identify_predcition.items()))
        #print('prediction.shape', prediction["quad_seg"].shape)
        prediction['center'] = index
        prediction['identify_seg'] = identify_predcition
        prediction['quad_id'] = torch.tensor([[0, 1, 2, 3]], device='cuda:0')
        prediction['seg_predcition']=seg_predcition['quad_seg']
        prediction['get_id'] = resized_tensor
        return prediction

    def tt(self,input):
        identify_ct = input

        seg_predcition = {}
        identify_predcition = self.quad_identify(identify_ct)

        prediction = dict(list(seg_predcition.items()) + list(identify_predcition.items()))
        return prediction


    def forward(self, input):
        prediction = self.forwardForTesting(input)
        return prediction

class SGAnetBuilder(object):
    def __init__(self, config):
        self.config = config
        self._quad_seg  = None
        self._quad_identify  = None
        self._network = None

    @property
    def quad_seg(self):
        if self._quad_seg is None:
            self._quad_seg = UNetBuilder(self.config["3dunet"]).network
        return self._quad_seg

    @property
    def quad_identify(self):
        if self._quad_identify is None:
            self._quad_identify = AttentionUNetBuilder(self.config["attentionUNet"]).network
        return self._quad_identify

    @property
    def network(self):
        if self._network is None:
            self._network = SGAnet(
                self.quad_seg,
                self.quad_identify,
                self.config
            )
        return self._network

def compute_predictions_from_features(network,X,targets,config):
    predictions = network(X)
    # volume = get_volume2(predictions,targets)  #后处理将预测转化为volume
    volume = get_volume(predictions,X,targets)  #后处理将预测转化为volume
    predictions["volume"] = volume
    return predictions

def get_pre(network,X,targets,config):
    predictions = network(X)
    return predictions


def train_on_batch(network, optimizer, loss_fn, metrics_fn, X, targets,
                   config):
    """Perform a forward and backward pass on a batch of samples and compute
    the loss.
    """
    optimizer.zero_grad()

    # Extract the per primitive features
    predictions  = compute_predictions_from_features(network,X,targets,config)

    # Do the forward pass to predict the primitive_parameters
    batch_loss = loss_fn(predictions, targets, config["loss"])

    # metrics_fn(predictions, targets)
    # Do the backpropagation
    batch_loss.backward()

    nn.utils.clip_grad_norm_(network.parameters(), 1)
    # Do the update
    optimizer.step()

    # StatsLogger.instance()["optimizer"].value = optimizer.optimizer.param_groups[0]["lr"]*10000
    return batch_loss.item()




def validate_on_batch(network, loss_fn, metrics_fn, X, targets, config):
    # Extract the per primitive features
    predictions  = compute_predictions_from_features(network,X,targets,config)

    # Do the forward pass to predict the primitive_parameters
    batch_loss = loss_fn(predictions, targets, config["loss"])
    # metrics_fn(predictions, targets)

    return batch_loss.item()

if __name__ == "__main__":
    from scripts.utils import load_config

    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/sga_net.yaml")
    builder = SGAnetBuilder(config)
    net = builder.network.cuda()


    net.load_state_dict(
        torch.load("/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/sga_net/model_00920")
    )

    net.quad_identify.attention.GBA.town = nn.Conv1d(8, 8, 1, 1)

    torch.save(
        net.state_dict(),
        "/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/sga_net/model_00930"
    )

