import os.path
from math import ceil, cos, pi, sin, sqrt

import igl
import torch
from itertools import combinations

try:
    import trimesh
except ImportError:
    trimesh = None

def uppool_mesh(vertices, faces):
    vertices,faces = uniform_unpool(vertices,faces,identical_face_batch = False)
    return vertices,faces


def get_commont_vertex(edge_pair):
    # 拿出两组边  edge_pair.shape = (N,2,2),

    a = edge_pair[:, 0] == edge_pair[:, 1]
    b = edge_pair[:, 0] == torch.flip(edge_pair[:, 1], dims=[1])

    return edge_pair[:, 0][a + b]


def one_ring(faces,V):
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
    return adj,L

#增加网格分辨率
def uniform_unpool(vertices_, faces_, identical_face_batch=True):
    if vertices_ is None:
        return None, None
    batch_size, _, _ = vertices_.shape
    new_faces_all = []
    new_vertices_all = []

    # 取出顶点和面
    for vertices, faces in zip(vertices_, faces_):
        face_count, _ = faces.shape
        # 原来顶点的数量
        vertices_count = len(vertices)
        # 对于ABC 三个顶点， 分别对应 BC , CA , AB 三条边
        edge_combinations_3 = torch.tensor([[1, 2], [2, 0], [0, 1]]).cuda()
        # 计算各条边  edges.shape = (N,3,2)
        half_edges = faces[:, edge_combinations_3]
        #将（n,m,2) ->(n*m,2)
        half_edges = half_edges.view(-1, 2)
        #对dim =1 排序，也就是边对顶点排序，（2,1) ->(1,2)
        half_edges_sort, _ = torch.sort(half_edges, dim=1)
        #unique_edge_indice是对应的重复的edge对应unique_edges中序号
        unique_edges, unique_edge_indices = torch.unique(half_edges_sort, return_inverse=True, dim=0)

        ''' Computer new vertices '''
        #边的中点生成新的顶点,edge_point
        #每条边连接两个顶点 p1,p2   ,face_edges = [p1,p2] ，并且对应两个顶点p3,p4,  edge2vert = [p3,p4]
        #这里有一个隐藏的关系，半边序号/3 ，取整为面的序号,
        ii = (half_edges[:,0] - half_edges[:,1])<0
        ii = ii.int()
        jj = unique_edge_indices
        idx = torch.stack([jj,ii ], dim=0)
        weight = torch.arange(face_count*3).cuda()
        # edge2half对于的半边，[顺序相反，顺序相同]
        edge2half = torch.sparse.FloatTensor(idx, weight, (int(face_count*3/2),2))
        edge2half = edge2half.to_dense()
        #edge2vert：边对应的顶点， 可以根据 faces 得出
        half2vert = faces.view(-1)
        edge2vert = half2vert[edge2half]
        edge_points = 3.0/8 * vertices[unique_edges].sum(dim=1) + 1.0/8 * vertices[edge2vert].sum(dim=1)

        # edge_adj_face = torch.zores(2,face_count*3/2)

        adj,L = one_ring(faces, vertices_count)
        targetVert = torch.matmul(L, vertices)
        #
        alpha = 5.0/8 - torch.pow(3.0/8 + 1.0/4 * torch.cos(2*torch.pi/adj),2)
        vertices = vertices * (1-alpha)+alpha*targetVert
        new_vertices = torch.cat([vertices, edge_points], dim=0)  # <----------------------- new vertices + old vertices
        new_vertices_all += [new_vertices[None]]

        ''' Compute new faces '''
        #生成新的面
        corner_faces = []

        half2edge = unique_edge_indices.view(-1,3) + vertices_count
        for j, combination in enumerate(edge_combinations_3):
            new_faces = half2edge.clone()
            new_faces[:,j] = faces[:,j]
            new_faces = torch.flip(new_faces,dims=[1])
            #新的面
            corner_faces += [new_faces]

        corner_faces = torch.cat(corner_faces, dim=0)
        new_faces_all += [torch.cat([corner_faces, half2edge], dim=0)[None]]  # new faces-3

        if identical_face_batch:
            new_vertices_all = new_vertices_all[0].repeat(batch_size, 1, 1)
            new_faces_all = new_faces_all[0].repeat(batch_size, 1, 1)
            break

    new_vertices_all = torch.cat(new_vertices_all,dim=0)
    new_faces_all = torch.flip(torch.cat(new_faces_all, dim=0),dims=[2])

    return new_vertices_all, new_faces_all


#增加网格分辨率,但是低GPU占用
def uniform_avg_center(vertices, faces):
    adj = torch.zeros(vertices.shape[0]).cuda()
    target_vertice = torch.zeros(vertices.shape).cuda()

    edge_combinations_3 = torch.tensor([1, 2, 0]).cuda()

    for i in range(3):
        adj.scatter_add_(0, faces[:,i],torch.ones(faces.shape[0]).cuda())
        target_vertice.scatter_add_(0, faces[:,i][:,None].repeat(1,3), vertices[faces[:,edge_combinations_3[i]]])
    adj = adj[:,None]
    target_vertice /= (1.0 * adj.repeat(1,3))

    return adj, target_vertice


# 增加网格分辨率,但是低GPU占用
def uniform_unpool_low_GPU_footprint(vertices_, faces_):
    if vertices_ is None:
        return None, None
    batch_size, _, _ = vertices_.shape
    new_faces_all = []
    new_vertices_all = []


    # 取出顶点和面
    for vertices, faces in zip(vertices_, faces_):
        face_count, _ = faces.shape
        # 原来顶点的数量
        vertices_count = len(vertices)
        # 对于ABC 三个顶点， 分别对应 BC , CA , AB 三条边
        edge_combinations_3 = torch.tensor([[1, 2], [2, 0], [0, 1]]).cuda()
        # 计算各条边  edges.shape = (N,3,2)
        half_edges = faces[:, edge_combinations_3]
        # 将（n,m,2) ->(n*m,2)
        half_edges = half_edges.view(-1, 2)
        # 对dim =1 排序，也就是边对顶点排序，（2,1) ->(1,2)
        half_edges_sort, _ = torch.sort(half_edges, dim=1)
        # unique_edge_indice是对应的重复的edge对应unique_edges中序号
        unique_edges, unique_edge_indices = torch.unique(half_edges_sort, return_inverse=True, dim=0)

        ''' Computer new vertices '''
        # 边的中点生成新的顶点,edge_point
        # 每条边连接两个顶点 p1,p2   ,face_edges = [p1,p2] ，并且对应两个顶点p3,p4,  edge2vert = [p3,p4]
        # 这里有一个隐藏的关系，半边序号/3 ，取整为面的序号,
        ii = (half_edges[:, 0] - half_edges[:, 1]) < 0
        ii = ii.int()
        jj = unique_edge_indices
        idx = torch.stack([jj, ii], dim=0)
        weight = torch.arange(face_count * 3).cuda()
        # edge2half对于的半边，[顺序相反，顺序相同]
        edge2half = torch.sparse.FloatTensor(idx, weight, (int(face_count * 3 / 2), 2))
        edge2half = edge2half.to_dense()
        # edge2vert：边对应的顶点， 可以根据 faces 得出
        half2vert = faces.view(-1)
        edge2vert = half2vert[edge2half]
        edge_points = 3.0 / 8 * vertices[unique_edges].sum(dim=1) + 1.0 / 8 * vertices[edge2vert].sum(dim=1)

        # edge_adj_face = torch.zores(2,face_count*3/2)

        adj, targetVert = uniform_avg_center(vertices, faces)

        #
        alpha = 5.0 / 8 - torch.pow(3.0 / 8 + 1.0 / 4 * torch.cos(2 * torch.pi / adj), 2)
        vertices = vertices * (1 - alpha) + alpha * targetVert
        new_vertices = torch.cat([vertices, edge_points], dim=0)  # <----------------------- new vertices + old vertices
        new_vertices_all += [new_vertices[None]]

        ''' Compute new faces '''
        # 生成新的面
        corner_faces = []

        half2edge = unique_edge_indices.view(-1, 3) + vertices_count
        for j, combination in enumerate(edge_combinations_3):
            new_faces = half2edge.clone()
            new_faces[:, j] = faces[:, j]
            new_faces = torch.flip(new_faces, dims=[1])
            # 新的面
            corner_faces += [new_faces]

        corner_faces = torch.cat(corner_faces, dim=0)
        new_faces_all += [torch.cat([corner_faces, half2edge], dim=0)[None]]  # new faces-3

    new_vertices_all = torch.cat(new_vertices_all, dim=0)
    new_faces_all = torch.cat(new_faces_all, dim=0)

    return new_vertices_all, new_faces_all

def sphere_mesh(n=512, radius=1.0, device="cpu"):
    vertexNum = [42,162,642,2562,10242,40962]
    num = 0
    for num in vertexNum:
        if num>=n:
            break
    vertices, _, _, faces, _, _ = igl.read_obj(os.path.join("../neural_parts_code/spheres","icosahedron_{}.obj".format(num)))
    vertices = torch.from_numpy(vertices).float().to(device)
    faces = torch.from_numpy(faces).long().to(device)
    return vertices * radius, faces



def sphere_edges(divisions=2, radius=1.0, device="cpu"):
    """Compute a sphere mesh using trimesh and return the sphere and the edges
    between points."""
    sphere = trimesh.creation.icosphere(subdivisions=divisions, radius=radius)
    edges = set()
    for face in sphere.faces:
        edges.add((face[0], face[1]))
        edges.add((face[0], face[2]))
        edges.add((face[1], face[2]))
    edges = torch.tensor(
        [[a, b] for a, b in edges],
        dtype=torch.long,
        device=device
    )
    vertices = torch.tensor(sphere.vertices, dtype=torch.float, device=device)

    return vertices, edges


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices

    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1]**2
    yy = quaternions[..., 2]**2
    zz = quaternions[..., 3]**2
    ww = quaternions[..., 0]**2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


def transform_to_primitives_centric_system(X, translations, rotation_angles):
    """
    Arguments:
    ----------
        X: Tensor with size BxNx3, containing the 3D points, where B is the
           batch size and N is the number of points
        translations: Tensor with size BxMx3, containing the translation
                      vectors for the M primitives
        rotation_angles: Tensor with size BxMx4 containing the 4 quaternion
                         values for the M primitives

    Returns:
    --------
        X_transformed: Tensor with size BxNxMx3 containing the N points
                       transformed in the M primitive centric coordinate
                       systems.
    """
    # Make sure that all tensors have the right shape
    assert X.shape[0] == translations.shape[0]
    assert translations.shape[0] == rotation_angles.shape[0]
    assert translations.shape[1] == rotation_angles.shape[1]
    assert X.shape[-1] == 3
    assert translations.shape[-1] == 3
    assert rotation_angles.shape[-1] == 4

    # Subtract the translation and get X_transformed with size BxNxMx3
    X_transformed = X.unsqueeze(2) - translations.unsqueeze(1)

    # R = euler_angles_to_rotation_matrices(rotation_angles.view(-1, 3)).view(
    R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
        rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
    )

    # Let as denote a point x_p in the primitive-centric coordinate system and
    # its corresponding point in the world coordinate system x_w. We denote the
    # transformation from the point in the world coordinate system to a point
    # in the primitive-centric coordinate system as x_p = R(x_w - t)
    X_transformed = R.unsqueeze(1).matmul(X_transformed.unsqueeze(-1))

    X_signs = (X_transformed > 0).float() * 2 - 1
    X_abs = X_transformed.abs()
    X_transformed = X_signs * torch.max(X_abs, X_abs.new_tensor(1e-5))

    return X_transformed.squeeze(-1)


def transform_to_world_coordinates_system(X_P, translations, rotation_angles):
    """
    Arguments:
    ----------
        X_P: Tensor with size BxMxSx3, containing the 3D points, where B is
              the batch size, M is the number of primitives and S is the number
              of points on each primitive-centric system
        translations: Tensor with size BxMx3, containing the translation
                      vectors for the M primitives
        rotation_angles: Tensor with size BxMx4 containing the 4 quaternion
                         values for the M primitives

    Returns:
    --------
        X_P_w: Tensor with size BxMxSx3 containing the N points
                transformed in the M primitive centric coordinate
                systems.
    """
    # Make sure that all tensors have the right shape
    assert X_P.shape[0] == translations.shape[0]
    assert translations.shape[0] == rotation_angles.shape[0]
    assert translations.shape[1] == rotation_angles.shape[1]
    assert X_P.shape[1] == translations.shape[1]
    assert X_P.shape[-1] == 3
    assert translations.shape[-1] == 3
    assert rotation_angles.shape[-1] == 4

    # Compute the rotation matrices to every primitive centric coordinate
    # system (R has size BxMx3x3)
    R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
        rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
    )
    # We need the R.T to get the rotation matrix from the primitive-centric
    # coordinate system to the world coordinate system.
    R_t = torch.einsum("...ij->...ji", (R,))
    assert R.shape == R_t.shape

    X_Pw = R_t.unsqueeze(2).matmul(X_P.unsqueeze(-1))
    X_Pw = X_Pw.squeeze(-1) + translations.unsqueeze(2)

    return X_Pw

if __name__ == "__main__":
    torch.cuda.set_device(2)

    new_verts, faces, mesh_class = torch.load('/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/all_teeth_0.pt', map_location=torch.device('cuda:2'))

    for i in range(4):
        new_verts, faces = uniform_unpool_low_GPU_footprint(new_verts, faces)