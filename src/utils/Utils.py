import torch
import torch.nn.functional
import torch.linalg

def normalize(tensor: torch.Tensor) -> torch.Tensor:

    if torch.where(tensor == 0).all():
        return tensor

    maxi = torch.max(tensor)
    mini = torch.min(tensor)

    return (tensor - mini) / (maxi - mini)

def matrix_normalize(tensor: torch.Tensor) -> torch.Tensor:
    
    if (tensor == 0).all():
        return tensor

    fro_norm = torch.linalg.norm(tensor, ord='fro')
    
    return tensor / fro_norm

def decimation(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    return tensor[::decim_row, ::decim_col].clone()

def decimation_adjoint(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    nb_row, nb_col = tensor.size()
    out = torch.zeros(
        size = (decim_row*nb_row, decim_col*nb_col),
        device = tensor.device
    )
    out[::decim_row, ::decim_col] = tensor.clone()
    return out

def decimation_v2(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    decimed = torch.nn.functional.avg_pool2d(
        input=tensor.unsqueeze(0),
        kernel_size=(decim_row, decim_col),
        stride=(decim_row, decim_col)
    )
    return decimed.squeeze(0)

def decimation_adjoint_v2(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor :
    res = torch.nn.functional.upsample(
        tensor.unsqueeze(0).unsqueeze(0),
        scale_factor=(decim_row, decim_col),
        mode='bicubic'
    )
    return res.squeeze(0).squeeze(0)
    
def decimation_v3(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:

    # Remove leading singleton dims until shape is 2D, 3D, or 4D
    while tensor.dim() > 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    original_dim = tensor.dim()

    # Normalize to [B,C,H,W]
    if original_dim == 2:          # [H,W]
        t = tensor.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:        # [C,H,W]
        t = tensor.unsqueeze(0)
    elif original_dim == 4:        # [B,C,H,W]
        t = tensor
    else:
        raise ValueError(f"decimation_v3: unsupported shape {tensor.shape}")

    # Apply decimation
    out = t[:, :, ::decim_row, ::decim_col].clone()

    # Restore original dim
    if original_dim == 2:
        return out.squeeze(0).squeeze(0)
    elif original_dim == 3:
        return out.squeeze(0)
    return out

    
    
def decimation_adjoint_v3(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:

    # Remove leading singleton dims until shape is 2D, 3D, or 4D
    while tensor.dim() > 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    original_dim = tensor.dim()

    # Normalize to [B,C,H,W]
    if original_dim == 2:          # [H,W]
        t = tensor.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:        # [C,H,W]
        t = tensor.unsqueeze(0)
    elif original_dim == 4:        # [B,C,H,W]
        t = tensor
    else:
        raise ValueError(f"decimation_adjoint_v3: unsupported shape {tensor.shape}")

    B, C, h, w = t.shape

    # Create upsampled output
    out = torch.zeros(B, C, h * decim_row, w * decim_col,
                      device=t.device, dtype=t.dtype)

    out[:, :, ::decim_row, ::decim_col] = t

    # Restore original dims
    if original_dim == 2:
        return out.squeeze(0).squeeze(0)
    elif original_dim == 3:
        return out.squeeze(0)
    return out



def multidimensional_soft(d: torch.Tensor, epsilon: float):
    """ Thresholding soft for multidimensional array
    Use generalization of sign function
    
    Params:
        - d : multidimensional array
        - epsilon : threshold

    Return:
        Array thresholded with dimesion equal to d
    """
    #print('d :', d.size())
    # l22 = 
    
    # s[s==0] = 
    #print('s :', s.size())
    s = torch.sqrt(torch.sum(d**2, axis=0))
    ss = torch.where(s > epsilon, (s-epsilon)/s, 0)
    output = torch.concat([(ss*d[i]).unsqueeze(0) for i in range(0, d.size()[0])], 0)
    #print('output :', output.size())
    #print(output.size())
    return output

def multidimensional_soft_v2(d: torch.Tensor, epsilon: float, gamma_zero: float=1e-12):
    """ Thresholding soft for multidimensional array
    Use generalization of sign function
    
    Params:
        - d : multidimensional array
        - epsilon : threshold
        - gamma_zero : for zero value (prevent "Error detected in DivBackward0")

    Return:
        Array thresholded with dimesion equal to d
    """
    #print('d :', d.size())
    # l22 = 
    
    # s[s==0] = 
    #print('s :', s.size())
    s = torch.sqrt(torch.sum(d**2, axis=0)+gamma_zero)

    ss = torch.where(s > epsilon, (s-epsilon)/s, 0)
    output = torch.concat([(ss*d[i]).unsqueeze(0) for i in range(0, d.size()[0])], 0)
    #print('output :', output.size())
    #print(output.size())
    return output

def soft(d: torch.Tensor, epsilon: float):
    """ Soft Thresholding 
    
    Params:
        - d : multidimensional array
        - epsilon : threshold

    Return:
        Array thresholded with dimesion equal to d
    """
    expr = torch.abs(d)-epsilon
    return torch.sign(d) * torch.where(0.0 < expr, expr, 0.0)


def dx(tensor: torch.Tensor) -> torch.Tensor:
    """ Derivation by column

    Params:
        - tensor
    
    Return:
        - first element of gradient
    """
    
    nb_rows, nb_cols = tensor.size()
    tensor_derivated = torch.zeros_like(tensor)

    tensor_derivated[:, 1:nb_cols] = \
        tensor[:, 1:nb_cols] - tensor[:, 0:nb_cols-1]

    tensor_derivated[:, 0] = tensor[:, 0] - tensor[:, nb_cols-1]

    return tensor_derivated
    

def dy(tensor: torch.Tensor) -> torch.Tensor:
    """ Derivation by line

    Params:
        - tensor
    
    Return:
        - second element of gradient
    """
    
    nb_rows, nb_cols = tensor.size()
    tensor_derivated = torch.zeros_like(tensor)
    
    tensor_derivated[1:nb_rows, :] = \
        tensor[1:nb_rows, :] - tensor[0:nb_rows-1, :]

    tensor_derivated[0, :] = tensor[0, :] - tensor[nb_rows-1, :]

    return tensor_derivated


def dxT(tensor: torch.Tensor) -> torch.Tensor:
    """ Derivation Transposed by column

    Params:
        - tensor
    
    Return:
        - first element of gradient transposed
    """
    
    #print(tensor.size())
    nb_rows, nb_cols = tensor.size()
    tensor_derivated = torch.zeros_like(tensor)
    
    tensor_derivated[:, 0:nb_cols-1] = \
        tensor[:, 0:nb_cols-1] - tensor[:, 1:nb_cols]

    tensor_derivated[:, nb_cols-1] = tensor[:, nb_cols-1] - tensor[:, 0]

    return tensor_derivated


def dyT(tensor: torch.Tensor) -> torch.Tensor:
    """ Derivation Transposed by line

    Params:
        - tensor
    
    Return:
        - second element of gradient transposed
    """
    nb_rows, nb_cols = tensor.size()
    tensor_derivated = torch.zeros_like(tensor)
    
    tensor_derivated[0:nb_rows-1, :] = \
        tensor[0:nb_rows-1, :] - tensor[1:nb_rows, :]

    tensor_derivated[nb_rows-1, :] = tensor[nb_rows-1, :] - tensor[0, :]

    return tensor_derivated

# def derivation(tensor: torch.Tensor, axis: int) -> torch.Tensor:
#     shape = tensor.size()
#     n = shape[axis]
#     taken = torch.take(tensor, axis=axis, indices=n-1)
#     shape[axis] = 1
#     to_add = torch.reshape(taken, shape)
#     return torch.diff(tensor, prepend=to_add, axis=axis)


def laplacian2D_v2(tensor: torch.Tensor) -> torch.Tensor:
    d_dx = dx(tensor)
    d_dy = dy(tensor)
    d2_d2x = dxT(d_dx)
    d2_d2y = dyT(d_dy)
    lap = d2_d2x + d2_d2y
    return lap



import sklearn.cluster
import numpy

def thresholding_kmeans(img: numpy.ndarray, k: int) -> numpy.ndarray:

    img_vectorized = numpy.reshape(img, newshape=(-1, 1), order='F')
    _, idx, _ = sklearn.cluster.k_means(img_vectorized, k, init='k-means++', random_state=0, n_init='auto')
    idx = numpy.reshape(idx, newshape=img.shape, order='F')

    cluster_mean = numpy.zeros(shape=k)

    for i in range(0, k):
        cluster_mean[i] = numpy.mean(img[idx==i])

    cluster_mean_sorted = numpy.sort(cluster_mean)

    thresholds = numpy.zeros(shape=k-1)

    for i in range(0, k-1):
        thresholds[i] = numpy.mean(cluster_mean_sorted[i:i+2])

    return thresholds

"""
def seg_result(
    img: numpy.ndarray, 
    ms_res: numpy.ndarray,
    thresholds: numpy.ndarray,
    k: int 
) -> numpy.ndarray:
    
    for i in range(0, k-1):

        #matplotlib.pyplot.imshow(img)

        if i == 0:
            temp = (ms_res < thresholds[i])
        elif i == k-1:
            temp = (thresholds[i-1] <= ms_res) \
                * (ms_res < thresholds[i])
            temp1 = (thresholds[i] <= ms_res)
        else:
            temp = (thresholds[i-1] <= ms_res) \
                * (ms_res < thresholds[i])
            
    #    matplotlib.pyplot.contour(temp, 'y')

    #if k > 2:
    #    matplotlib.pyplot.imshow(img)
    #    matplotlib.pyplot.contour(temp1, 'y')
    


    seg = numpy.zeros_like(img)

    for i in range(0, k-1):

        if i == 0:
            temp = ms_res < thresholds[i]
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
            if k == 2:
                temp = thresholds[i] <= ms_res
                seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
        elif i == k-1:
            temp = (thresholds[i] <= ms_res) \
                * (ms_res < thresholds[i])
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
            temp = (thresholds[i] <= ms_res)
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
        else:
            temp = (thresholds[i-1] <= ms_res) \
                * (ms_res < thresholds[i])
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)

    return seg
    #matplotlib.pyplot.imshow(seg)
"""

def seg_result(
    ms_res: numpy.ndarray,
    thresholds: numpy.ndarray,
    k: int 
) -> numpy.ndarray:

    seg = numpy.zeros_like(ms_res)

    for i in range(0, k-1):

        if i == 0:
            temp = ms_res < thresholds[i]
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
            if k == 2:
                temp = thresholds[i] <= ms_res
                seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
        elif i == k-1:
            temp = (thresholds[i] <= ms_res) \
                * (ms_res < thresholds[i])
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
            temp = (thresholds[i] <= ms_res)
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
        else:
            temp = (thresholds[i-1] <= ms_res) \
                * (ms_res < thresholds[i])
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)

    return seg
    #matplotlib.pyplot.imshow(seg)