import numpy as np
import cv2
import scipy.signal
import torch
def torch_reshape(input, shape):
    """
    :param input:  N-way tensor
    :param shape:  list
    :return:  the reshape tensor of the shape
    """
    input_index = [len(input.shape)-i for i in range(1, len(input.shape)+1)]
    input = input.permute(input_index)
    output_index = [len(shape)-i for i in range(1, len(shape)+1)]
    shape = list(reversed(shape))
    output = torch.reshape(input, shape)
    return output.permute(output_index)


def np_reshape(input, shape):
    '''
    input: numpy array
    shape: a list of int
    return: numpy array
    '''
    s = input.shape
    input = np.transpose(input)
    shape = list(map(int,list(reversed(shape))))
    ans = np.reshape(input, shape)
    ans = ans.transpose()
    return ans

def crmask(size, rate, mode='random'):
    global mask
    x = np.zeros(size)
    if mode == 'random':
        N = np.size(x)
        ind = np.random.choice(N, round(rate * N), replace=False)
        mask = np_reshape(x, [N, 1])
        mask[ind] = 1
    elif mode == 'tube':
        mask = np_reshape(x,[size[0]*size[1], size[2]])
        ind = np.random.choice(size[0]*size[1], round(rate*size[0]*size[1]), replace=False)
        mask[ind, :] = 1
    elif mode == 'slice':
        mask = x
        ind = np.random.choice(size[2], round(rate*size[2]), replace=False)
        mask[:,:,ind] = 1

    mask = np_reshape(mask, size)
    return mask

def _as_floats(im1, im2):
    """Promote im1, im2 to nearest appropriate floating point precision."""
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2

def _assert_compatible(im1, im2):
    """Raise an error if the shape and dtype do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return

def compare_mse(im1, im2):
    """Compute the mean-squared error between two images.

    Parameters
    ----------
    im1, im2 : ndarray
        Image.  Any dimensionality.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    """
    _assert_compatible(im1, im2)
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)

def compare_psnr(im_true, im_test, data_range=None):

    psnr = 0
    k = im_true.shape[1]
    for jj in range(k):
        psnr = psnr + 10 * np.log10((1 ** 2) / compare_mse(im_true[ :, jj], im_test[ :, jj]))

    return psnr / k

def ssim_index(img1, img2, K=(0.01, 0.03), window=np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T), L=255):
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    window = window / np.sum(window, axis=(0,1))
    mu1 = scipy.signal.correlate2d(img1, window, 'valid')
    mu2 = scipy.signal.correlate2d(img2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = scipy.signal.correlate2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = scipy.signal.correlate2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = scipy.signal.correlate2d(img1 * img2, window, 'valid') - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq +C1) * (sigma1_sq + sigma2_sq +C2))
    else:
        numerator1 = 2 * mu1_mu2 +C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape)
        index = denominator1 * denominator2 > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) and (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)

    return mssim

def compare_ssim(img1, img2):
    ssim = []
    ssim.append(ssim_index(img1*255, img2*255))

    ssim = np.mean(ssim)
    return ssim
