import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.backends.cudnn as cudnn

MODE_ZEROS = 0
MODE_BORDER = 1


def affine_grid(theta, size):
    return AffineGridGenerator.apply(theta, size)


# TODO: Port these completely into C++
class AffineGridGenerator(Function):

    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        N, C, D, H, W = size
        ctx.size = size

        #ctx.is_cuda = True

        base_grid = theta.new(N, D, H, W, 4)

        w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
        h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
        d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

        base_grid[:, :, :, :, 0] = w_points
        base_grid[:, :, :, :, 1] = h_points
        base_grid[:, :, :, :, 2] = d_points
        base_grid[:, :, :, :, 3] = 1
        ctx.base_grid = base_grid
        grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
        grid = grid.view(N, D, H, W, 3)
        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        N, C, D, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, D, H, W, 3])
        base_grid = ctx.base_grid
        grad_theta = torch.bmm(
            base_grid.view(N, D * H * W, 4).transpose(1, 2),
            grad_grid.view(N, D * H * W, 3))
        grad_theta = grad_theta.transpose(1, 2)
        return grad_theta, None


if __name__ == "__main__":

    from scipy.spatial.transform import Rotation as R
    import torch.nn.functional as F

    theta_rotate = torch.zeros((4,4))
    theta_rotate[0, 2] = 1
    theta_rotate[1, 0] = 1
    theta_rotate[2, 1] = 1
    theta_rotate[3, 3] = 1


    # a = R.random().as_matrix()
    # for i in range(3):
    #     for j in range(3):
    #         theta_rotate[i,j] = a[i,j]

    x = torch.ones((1,1,3,3,3))

    x1 = torch.ones((1, 1, 3, 3, 3))
    x2 = torch.ones((1, 1, 3, 3, 3))
    x3 = torch.ones((1, 1, 3, 3, 3))

    point = []
    point_mat = torch.ones((1,1,3,3,3,3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                point.append(torch.tensor([i-1.0, j-1.0, k-1.0,1.0]))
                point_mat[0, 0, i, j, k] = point[i*9+j*3+k][:3]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                x1[0,0,i,j,k]  =  i - 1
                x2[0, 0, i, j, k] = j -1
                x3[0, 0, i, j, k] = k- 1

    theta_rotate = theta_rotate[0:3, :].view(-1, 3, 4)
    grid = affine_grid(theta_rotate, x.shape)


    x1 = F.grid_sample(x1, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    x2 = F.grid_sample(x2, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    x3 = F.grid_sample(x3, grid, mode='nearest', padding_mode='zeros', align_corners=True)

    x = torch.stack([x1, x2, x3], 5)


    for i in range(3):
        for j in range(3):
            for k in range(3):
                print("{} {} {} to ".format(i,j,k))
                print(x[0, 0, i, j, k]+1)

    theta_rotate = torch.zeros((4, 4))
    # theta = torch.ones((4, 4))
    #
    theta_rotate[0, 2] = 1
    theta_rotate[1, 0] = 1
    theta_rotate[2, 1] = 1
    theta_rotate[3, 3] = 1

    theta_rotate = theta_rotate.t()
    theta = theta_rotate[:,:3]
    print("ddddddddddddddddddddddddddddddddd")
    for i in range(3):
        for j in range(3):
            for k in range(3):
                print("{} {} {} to ".format(i, j, k))
                print(point[i * 9 + j * 3 + k] @ theta + 1)
                # print(point[i * 9 + j * 3 + k] + 1)

