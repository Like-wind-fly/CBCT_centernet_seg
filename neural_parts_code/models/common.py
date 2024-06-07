#逐像素的实例分类器 ：Instance-aware Point Classifier，我们这里同样叫IPC
#可以分成以下模块
# 根据坐标信息在特征图上采样的模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_parts_code.stats_logger import StatsLogger
from neural_parts_code.models.ResUnetEncode import ResUNetEncode
from neural_parts_code.models.ResUnetDecode import ResUNetDecode
from pytorch3d.ops import cot_laplacian
from scripts.common import write_point

# feature_sampling: 这里使用最简单的特征提取，双线性插值
class BasicSample(nn.Module):

    def __init__(self, features_count):
        super(BasicSample, self).__init__()

    #输入vertices.shape = B,N,3
    # voxel_features = B,C,D,H,W
    def forward(self, voxel_features, vertices):
        neighbourhood = vertices[:, :, None, None]
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border',
                                 align_corners=True)
        features = features[:, :, :, 0, 0].transpose(2, 1)
        return features


def mask_head(mask_feat,controllers_pred, dim = 2):
    # for CondInst


    # for each image

    controllers = controllers_pred
    ins_num = controllers.shape[0]

    weights1 = controllers[:, :16].reshape(-1, 4, 4).reshape(-1, 4)
    bias1 = controllers[:, 16:20].flatten()
    weights2 = controllers[:, 20:36].reshape(-1, 4, 4).reshape(-1, 4)
    bias2 = controllers[:, 36:40].flatten()
    weights3 = controllers[:, 40:44].reshape(-1, 1, 4).reshape(-1, 4)
    bias3 = controllers[:, 44:45].flatten()

    if dim == 1:
        mask_feat = mask_feat.transpose(2, 1)
        B,C,N  = mask_feat.shape

        mask_feat = mask_feat.reshape(1,B*C,N)
        weights1 = weights1.unsqueeze(-1)
        weights2 = weights2.unsqueeze(-1)
        weights3 = weights3.unsqueeze(-1)

        conv1 = F.conv1d(mask_feat, weights1, bias1 , groups=ins_num).relu()
        conv2 = F.conv1d(conv1, weights2, bias2, groups=ins_num).relu()
        masks_per_image = F.conv1d(conv2, weights3, bias3, groups=ins_num)[0].sigmoid()
        masks_per_image = masks_per_image.unsqueeze(-1)
        return masks_per_image

    if dim == 2:
        N,C,D,W = mask_feat.shape
        mask_feat = mask_feat.reshape(1, N * C, D , W)

        weights1 = weights1.unsqueeze(-1).unsqueeze(-1)
        weights2 = weights2.unsqueeze(-1).unsqueeze(-1)
        weights3 = weights3.unsqueeze(-1).unsqueeze(-1)

        conv1 = F.conv2d(mask_feat, weights1, bias1).relu()
        conv2 = F.conv2d(conv1, weights2, bias2, groups=ins_num).relu()
        masks_per_image = F.conv2d(conv2, weights3, bias3, groups=ins_num)[0].sigmoid()

        return masks_per_image

    if dim == 3:
        N,C,D,W,H = mask_feat.shape
        mask_feat = mask_feat.reshape(1, N * C, D, W, H)

        weights1 = weights1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights2 = weights2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights3 = weights3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        conv1 = F.conv3d(mask_feat, weights1, bias1, groups=ins_num).relu()
        conv2 = F.conv3d(conv1, weights2, bias2, groups=ins_num).relu()
        masks_per_image = F.conv3d(conv2, weights3, bias3, groups=ins_num)[0].sigmoid()
        masks_per_image = masks_per_image.unsqueeze(1)
        return masks_per_image

def normalize(tensor):
    if len(tensor.shape) == 2:
        length = tensor.norm(dim = 1,keepdim=True)
        return  1.0 * tensor / length
    elif len(tensor.shape) == 3:
        length = tensor.norm(dim = 2,keepdim=True)
        return  1.0 * tensor / length

def getNormal(verts, faces):
    # faces.shape = D,F,3 , but faces[i,:,:] is same
    D,V,_ = verts.shape
    D,F,_ = faces.shape

    faces = faces[0]

    # A.shape = D,F,3
    A = verts[:, faces[:,0],:].resize(D*F,3)
    B = verts[:, faces[:,1], :].resize(D*F,3)
    C = verts[:, faces[:,2], :].resize(D*F,3)

    AB = normalize(B - A)
    AC = normalize(C - A)
    BC = normalize(C - B)
    # vmax = verts.max()
    # max = A.max()
    # max = AB.max()

    A_angle = torch.acos((AB * AC).sum(dim=1))
    B_angle = torch.acos((-1*AB * BC).sum(dim=1))
    C_angle = torch.acos((AC * BC).sum(dim=1))

    N = torch.cross( AB,AC, dim=1)
    N = normalize(N)

    N = N.resize(D, F, 3)

    # A_angle.shape = D,F

    A_angle = A_angle.resize(D, F, 1)
    B_angle = B_angle.resize(D, F, 1)
    C_angle = C_angle.resize(D, F, 1)

    #weight.shape = D,F,3
    weight = torch.stack([A_angle,B_angle,C_angle],dim=-1)
    weight = weight.view(-1)

    kk = torch.arange(D)[:,None,None].repeat(1,F,3).cuda()
    ii = faces[None,:,:].repeat(D,1,1)
    jj = torch.arange(0,F)[None,:,None].repeat(D,1,3).cuda()

    idx = torch.stack([kk,ii, jj], dim=0).view(3, D* F * 3)

    #vertex_face_adjacency.shape = D,V,F   N.shape =D, F, 3
    vertex_face_adjacency = torch.sparse.FloatTensor(idx, weight , (D,V,F))
    vertex_face_adjacency = vertex_face_adjacency.to_dense()
    # vertex_face_adjacency.shape = D,V,F ,

    normal = torch.einsum('dvf,dfi->dvi',vertex_face_adjacency,N)
    normal = normalize(normal)
    # Nm = normal.max()
    return normal


def uniform_smooth(verts,faces,factor = 0.5, step = 1):
    if len(faces.shape) ==3:
        faces = faces[0]

    B,V,_ = verts.shape
    F, _ = faces.shape

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    one = torch.ones(F*3).cuda()
    L = torch.sparse.FloatTensor(idx, one, (V, V))
    L += L.t()
    L = L.to_dense()
    adj = L.sum(dim=1,keepdim = True)
    L/=adj

    for i in range(step):
        targetVert = torch.matmul(L,verts)
        moveVector = targetVert - verts
        verts += moveVector*factor

    return verts


# IPC作为顶点分类器 ，
# UNet 基础作为backbone
# controller , 用来生成实例 mask FCN Head 的参数, decode.output的输出为作为输入
# controller town 维度不增加的前提下
class IPC(nn.Module):
    def __init__(self,config):
        super(IPC, self).__init__()
        self.config = config
        # self.encode = ResUNetEncode(config["feature_extractor"], training=True)
        # self.decode = ResUNetDecode(config["feature_extractor"], training=True)
        self.encode = None
        self.decode = None
        self.controller = nn.Linear(config["feature_extractor"]["feature_size"],config["IPC"]["feature_size"])
        self.mask_tower = nn.Sequential(
            nn.Conv3d(config["feature_extractor"]["first_layer_channels"],config["IPC"]["controller_hidden_size"], 3, 1, padding=1),
            nn.PReLU(config["IPC"]["controller_hidden_size"])
        )
        self.sample = BasicSample(config["IPC"]["controller_hidden_size"])

    def onlyTrainController(self,yes_or_no):
        if yes_or_no == True:
            self.encode.requires_grad = False
            self.decode.requires_grad = False
        else :
            self.encode.requires_grad = True
            self.decode.requires_grad = True

    def get_mask_controler(self, x):
        feature = self.encode(x)
        controller_feature = self.controller(feature)
        self.decode(self.encode.output)
        output = self.decode.output
        mask = output[-1]
        mask = self.mask_tower(mask)
        return mask,controller_feature

    def get_step(self,select_point_ipc):
        select_point_ipc = select_point_ipc.squeeze(dim = 3)
        B,M,S = select_point_ipc.shape

        sign_ipc = select_point_ipc.clone()
        sign_ipc[sign_ipc < 0] = -1
        sign_ipc[sign_ipc > 0] = 1

        subMax = torch.zeros([S,S])
        for i in range(S):
            subMax[i,i] = 1
            if i !=S-1:
                subMax[i+1,i] = -1

        subMax = subMax.cuda()

        # changeSign = B,M,S
        sign_ipc = sign_ipc.resize(B*M,S)
        changeSign = torch.mm(sign_ipc,subMax).squeeze()
        changeSign[:,S-1] = 1
        changeSign[changeSign!=0] = 1

        # idx = [S ... 3,2,1]
        idx = reversed(torch.Tensor(range(1,S+1))).cuda()

        # temp = B,M,S
        temp = torch.einsum("ab,b->ab", (changeSign, idx))
        # step.shape = B,M
        step = torch.argmax(temp, dim=1)

        a_step = step.unsqueeze(dim = -1)
        a_step = torch.cat([a_step,a_step+1],dim= 1)

        select_point_ipc = select_point_ipc.resize(B*M,S)
        add_ipc = -1 * select_point_ipc[:, S - 1].unsqueeze(dim = -1)

        select_point_ipc = torch.cat([select_point_ipc, add_ipc ], dim=1)
        a = torch.gather(select_point_ipc, 1, a_step)
        diff = a[:, 0] - a[:, 1]
        add_step = a[:,0]/(diff)
        diff = diff.abs()
        add_step[diff<1e-5] = 0.5

        step = add_step + step
        step = step.resize(B,M)
        return step




    def deform(self,verts,faces,mask,controller_feature,deformable = None):
        # verts.shape = B,V,3
        # faces.shape = B,F,3
        B = verts.shape[0]
        V = verts.shape[1]

        # if deformable == None:
        #     deformable = torch.ones(V).bool().cuda()

        direction = torch.ones([B,V,1]).cuda()

        local_verts = verts.clone()

        ipc = self.classification_point(local_verts,mask,controller_feature)


        local_direction = ipc.clone() - 0.5
        local_direction[local_direction<0] = -1
        local_direction[local_direction>0] = 1
        direction = local_direction

        # normal.shape = B,V,3
        normals = getNormal(verts, faces)
        # nmax = normals.max()

        deformVert = verts
        deformNormal = normals
        direction = direction

        step_number = self.config["IPC"]["step_number"]
        step_size = self.config["IPC"]["step_size"]

        deformNormal = direction * deformNormal

        select_point = [(i*step_size*deformNormal + deformVert) for i in range(1,step_number)]
        select_point = torch.stack(select_point, dim=2)
        B,M,S,D = select_point.shape
        # step.shape = B,M*step_number,3
        select_point = select_point.resize(B,M*S,D)

        select_point_ipc = self.classification_point(select_point, mask, controller_feature)


        select_point_ipc = select_point_ipc.resize(B, M, S, 1)

        ipc = ipc.unsqueeze(dim=-1)

        select_point_ipc = torch.cat([ipc,select_point_ipc],dim=2)

        # select_point = torch.cat([deformVert.unsqueeze(dim=2),select_point.resize(B,M,S,3)],dim=2)
        # torch.save(
        #     [select_point.resize(B,M*(S+1),3), select_point_ipc.resize(B,M*(S+1),1), deformVert, ipc],
        #      "prediction.pt"
        # )
        select_point_ipc = select_point_ipc - 0.5

        #step B,M , deformNormal: B,M,3
        step = self.get_step(select_point_ipc).unsqueeze(-1)
        # ms = step.max()
        # md = deformNormal.max()

        moveToPoint = step_size*step*deformNormal

        moveToPoint = uniform_smooth(moveToPoint, faces, factor=0.05, step=2)
        moveToPoint = moveToPoint+verts

        verts = moveToPoint
        # vm = verts.max()
        verts = uniform_smooth(verts, faces, factor=0.01, step=1)

        return verts


    def classification_grad(self,mask,controller_feature):
        pred = {}
        volume = mask_head(mask,controller_feature,dim=3)
        # pred["volume"] = volume
        return volume

    def classification_point(self, point, mask, controller_feature):
        point_feature = self.sample(mask,point)
        pcl = mask_head(point_feature, controller_feature, dim=1)
        point = point.abs()
        maxCo,_ = point.max(dim =2,keepdim = True)
        pcl[maxCo>1] = 0
        # pred["point_class"] = point
        return pcl


class IPCBuilder(object):
    def __init__(self, config):
        self.config = config
        self._network = None
        self.training = True

    @property
    def network(self):
        if self._network is None:
            self._network = IPC(self.config)
            self._network.encode = ResUNetEncode(self.config["feature_extractor"], training=True)
            self._network.decode = ResUNetDecode(self.config["feature_extractor"], training=True)
        return self._network

def compute_predictions_from_features(network, mask, controller_feature,targets,config):
    # Dictionary to keep the predictions
    predictions = {}
    if config["network"].get("point_class", False):
        # pred per point class , point is cube point in target[0]
        point = targets[0].squeeze(2)
        point_class = network.classification_point(point, mask, controller_feature)
        predictions["point_class"] = point_class

    if config["network"].get("volume", False):
        # pred per point class , point is cube point in target[0]
        volume = network.classification_grad( mask, controller_feature)
        predictions["volume"] = volume

    return predictions

def train_on_batch(network, optimizer, loss_fn, metrics_fn, X, targets,
                   config):
    """Perform a forward and backward pass on a batch of samples and compute
    the loss.
    """
    optimizer.zero_grad()

    # Extract the per primitive features

    mask, controller_feature = network.get_mask_controler(X)

    predictions  = compute_predictions_from_features(network,mask, controller_feature, targets,config)

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

    mask, controller_feature = network.get_mask_controler(X)
    predictions  = compute_predictions_from_features(network,mask, controller_feature, targets,config)

    # Do the forward pass to predict the primitive_parameters
    batch_loss = loss_fn(predictions, targets, config["loss"])
    metrics_fn(predictions, targets)

    return batch_loss.item()




if __name__ == "__main__":
    import torch
    import os
    from scripts.utils import load_config
    from ResUnetEncode import ResUNetEncode
    from ResUnetDecode import ResUNetDecode
    from torch.nn.parameter import Parameter
    from mappingUnet import MappingUnetBuilder

    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/mappingNet01.yaml")

    device = torch.device("cuda:{}".format(config["network"]["cuda"]))


    netbuild = MappingUnetBuilder(config)
    unet = netbuild.network
    unet.load_state_dict(torch.load("/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/mappingUnet2/model_02990",map_location='cuda:1'))
    os.environ["CUDA_VISIBLE_DEVICES"] = config["network"]["cuda"]


    x = torch.rand(3, 2, 64, 64, 64).cuda()
    point = torch.rand(3,500,3).cuda()
    ipc = IPC(config).cuda()
    ipc.encode = unet.feature_extractor
    ipc.decode = unet.refine_decode

    torch.save(
        ipc.state_dict(),
        os.path.join("/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/IPC", "model_00000")
    )
    pass