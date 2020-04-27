import torch


_matrix = torch.tensor([[0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]])


def rgb2lab(rgb, shift=False):
    # rgb NxCxHxW
    matrix = _matrix.to(rgb.device)

    rgb = rgb.permute(0, 2, 3, 1)
    rgb_shape = rgb.shape
    rgb = rgb.contiguous().view(-1, rgb_shape[-1])

    rgb = (rgb > 0.04045).float() * ((rgb + 0.055) / 1.055) ** 2.4 + (rgb <= 0.04045).float() * rgb / 12.92
    rgb *= 100

    xyz = torch.matmul(rgb, matrix)
    xyz[:, 0] /= 95.047
    xyz[:, 1] /= 100.0
    xyz[:, 2] /= 108.883

    xyz = (xyz > 0.008856).float() * torch.pow(xyz, 1 / 3.0) + (xyz <= 0.008856).float() * (
                (7.787 * xyz) + (16 / 116))

    L = 116 * xyz[:, 1] - 16
    a = 500 * (xyz[:, 0] - xyz[:, 1])
    b = 200 * (xyz[:, 1] - xyz[:, 2])

    if shift:
        L *= 2.55
        a += 128
        b += 128


    lab = torch.stack((L, a, b), 1)

    lab = lab.view(rgb_shape).contiguous().permute(0, 3, 1, 2)

    return lab


def squared_error(imgs1, imgs2, normalize=True):
    lab1 = rgb2lab(imgs1, shift=True)
    lab2 = rgb2lab(imgs2, shift=True)
    norm = 255 if normalize else 1
    diff = torch.pow((lab1 - lab2) / norm, 2).sum(1, keepdim=True) / 3  # .permute(0, 3, 1, 2)
    return diff