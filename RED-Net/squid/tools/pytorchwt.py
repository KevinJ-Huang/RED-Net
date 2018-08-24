# Pytorch based multilevel wavelet decomposition and reconstruction


"""
Provides:
    1. wavelet_n_dec(x, level): multilevel 2D wavelet (haar) decompostion
    2. wavelet_dec(x): one level 2D wavelet (haar) decomposition
    3. wavelet_n_rec(y): multilevel 2D wavelet (haar) reconstruction
    4. wavelet_rec(x_LL, x_LH, x_HL, x_HH): one level 2D wavelet (haar) reconstruction

"""

import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable


class pytorchwt(object):
    """
    Haar 2D wavelet decomposition and reconstruction

    """
    def __init__(self, gpu_id=0, is_cpu=False, feature_num=3):
        # Get Haar filter
        inver_sqrt2 = 1.0 / math.sqrt(2)
        self.feature_num = feature_num

        self.h0 = Variable(torch.FloatTensor([[inver_sqrt2, inver_sqrt2]]).repeat(self.feature_num, 1).
                           resize_(self.feature_num, 1, 1, 2))
        self.h1 = Variable(torch.FloatTensor([[inver_sqrt2, -inver_sqrt2]]).repeat(self.feature_num, 1).
                           resize_(self.feature_num, 1, 1, 2))

        self.h0_t = Variable(torch.FloatTensor([[inver_sqrt2], [inver_sqrt2]]).repeat(self.feature_num, 1).
                             resize_(self.feature_num, 1, 2, 1))
        self.h1_t = Variable(torch.FloatTensor([[inver_sqrt2], [-inver_sqrt2]]).repeat(self.feature_num, 1).
                             resize_(self.feature_num, 1, 2, 1))

        if not is_cpu:
            self.h0 = self.h0.cuda(gpu_id)
            self.h1 = self.h1.cuda(gpu_id)
            self.h0_t = self.h0_t.cuda(gpu_id)
            self.h1_t = self.h1_t.cuda(gpu_id)

    def wavelet_n_dec(self, x, level=3):
        """
        Haar 2D multiple level wavelet decomposition

        :param x: input N * C * W * H tensor, where C = self.feature_num
        :param level: decomposition level
        :return: y: a list of subbands
        """
        if level <= 0:
            return [x]

        xlo, xLH, xHL, xHH = self.wavelet_dec(x)
        xhi_dir = [xLH, xHL, xHH]

        # Recursively call on low band
        y = self.wavelet_n_dec(xlo, level=level - 1)

        # add subbands to the final output
        y.append(xhi_dir)

        return y

    def wavelet_dec(self, x):
        """
        Haar 2D one level wavelet decomposition

        :param x: input N * C * W * H tensor, where C = self.feature_num
        :return: x_LL, xLH, x_HL, x_HH: Four 2-D wavelet subbands, each is a four-dimension tensor
        """

        # Row-wise filtering
        x_L = F.conv2d(x, self.h0, stride=(1, 2), groups=self.feature_num)

        x_H = F.conv2d(x, self.h1, stride=(1, 2), groups=self.feature_num)

        # Column wise filtering
        x_LL = F.conv2d(x_L, self.h0_t, stride=(2, 1), groups=self.feature_num)
        x_LH = F.conv2d(x_L, self.h1_t, stride=(2, 1), groups=self.feature_num)
        x_HL = F.conv2d(x_H, self.h0_t, stride=(2, 1), groups=self.feature_num)
        x_HH = F.conv2d(x_H, self.h1_t, stride=(2, 1), groups=self.feature_num)

        return x_LL, x_LH, x_HL, x_HH

    def wavelet_n_rec(self, y):
        """
        Haar 2D multiple level wavelet reconstruction

        :param y: a list of subbands
        :return: x: reconstructed features or image
        """
        n = len(y) - 1
        if n <= 0:
            return y[0]

        xlo = self.wavelet_n_rec(y[:-1])

        x = self.wavelet_rec(xlo, y[-1][0], y[-1][1], y[-1][2])

        return x

    def wavelet_rec(self, x_LL, x_LH, x_HL, x_HH):
        """
        Haar 2D one level wavelet reconstruction

        :param x_LL, x_LH, x_HL, x_HH: input subbands, each is a four-dimension tensor
        :return: x: reconstructed features or image
        """
        # Column-wise filtering
        x_L = F.conv_transpose2d(x_LL, self.h0_t, stride=(2, 1), groups=self.feature_num)
        x_L = x_L + F.conv_transpose2d(x_LH, self.h1_t, stride=(2, 1), groups=self.feature_num)

        x_H = F.conv_transpose2d(x_HL, self.h0_t, stride=(2, 1), groups=self.feature_num)
        x_H = x_H + F.conv_transpose2d(x_HH, self.h1_t, stride=(2, 1), groups=self.feature_num)

        # Row-wise filtering
        x = F.conv_transpose2d(x_L, self.h0, stride=(1, 2), groups=self.feature_num)
        x = x + F.conv_transpose2d(x_H, self.h1, stride=(1, 2), groups=self.feature_num)

        return x


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from time import clock

    #import pywt

    # Read test image and show
    img_path = '/3T/zam/image_restoration/image_restoration_MTLAB_PyTorch/test_data/lena.png'
    img = Image.open(img_path).convert("RGB")
    #img.show(title='Original Image')

    # get numpy array
    img_arr = np.array(img)

    test_times = 100

    """ Test pywt """
    """
    start_pywt = clock()

    for ind in range(test_times):
        print "pywt: processing count {} ...".format(ind + 1)

        y_pywt_dec_0 = pywt.wavedec2(img_arr[:, :, 0], 'haar', level=3)
        y_pywt_dec_1 = pywt.wavedec2(img_arr[:, :, 1], 'haar', level=3)
        y_pywt_dec_2 = pywt.wavedec2(img_arr[:, :, 2], 'haar', level=3)

        x_pywt_rec_0 = pywt.waverec2(y_pywt_dec_0, 'haar')
        x_pywt_rec_1 = pywt.waverec2(y_pywt_dec_1, 'haar')
        x_pywt_rec_2 = pywt.waverec2(y_pywt_dec_2, 'haar')

    time_pywt = clock() - start_pywt
    print "Average processing time for pywt is {}ms".format(1000.0 * time_pywt / test_times)

    img_pywt = Image.fromarray(np.stack([x_pywt_rec_0, x_pywt_rec_1, x_pywt_rec_2], axis=2).astype(np.uint8))
    img_pywt.show(title='Wavelet Reconstructed Image')
    """

    """ Test pytorchwt """
    img_tensor = Variable(torch.from_numpy(np.expand_dims(img_arr.transpose((2, 0, 1)), axis=0)).float()).cuda()

    start_pytorchwt = clock()

    pytorch_wt = pytorchwt(feature_num=3)

    for ind in range(test_times):
        print "pytorchwt: processing count {} ...".format(ind + 1)

        y_pytorchwt = pytorch_wt.wavelet_n_dec(img_tensor, level=3)
        x_pytorchwt = pytorch_wt.wavelet_n_rec(y_pytorchwt)

    time_pytorchwt = clock() - start_pytorchwt
    print "Average processing time for pytorchwt is {}ms".format(1000.0 * time_pytorchwt / test_times)

    img_pytorchwt = Image.fromarray(x_pytorchwt.data.cpu().numpy().astype(np.uint8).squeeze().transpose((1, 2, 0)))
    #img_pytorchwt.show(title='Wavelet Reconstructed Image')

    error = np.mean(np.fabs(img_arr - np.array(img_pytorchwt)))
    print "Reconstruction error: {}".format(error)
