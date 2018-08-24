import math
import torch
import torch.nn as nn
import time


class Red20ResiDownsample(nn.Module):
    def __init__(self, image_c=3, feature_num=64, downsample_location=[0]):
        super(Red20ResiDownsample, self).__init__()
        self.feature_num = feature_num

        is_downsample = 21 * [False]
        for ind in downsample_location:
            is_downsample[ind] = True

        self.conv1 = self.make_conv_relu(feature_num_in=image_c, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[1])
        self.conv2 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[2])
        self.conv3 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[3])
        self.conv4 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[4])
        self.conv5 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[5])
        self.conv6 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[6])
        self.conv7 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[7])
        self.conv8 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[8])
        self.conv9 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                         is_downsampling=is_downsample[9])
        self.conv10 = self.make_conv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                          is_downsampling=is_downsample[10])

        self.deconv1 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[11])
        self.deconv2 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[12])
        self.deconv3 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[13])
        self.deconv4 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[14])
        self.deconv5 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[15])
        self.deconv6 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[16])
        self.deconv7 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[17])
        self.deconv8 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[18])
        self.deconv9 = self.make_deconv_relu(feature_num_in=self.feature_num, feature_num_out=self.feature_num,
                                             is_upsampling=is_downsample[19])
        if is_downsample[20]:
            self.deconv10 = nn.ConvTranspose2d(self.feature_num, image_c, kernel_size=4, stride=2,
                                               padding=1, bias=True)
        else:
            self.deconv10 = nn.ConvTranspose2d(self.feature_num, image_c, kernel_size=3, stride=1,
                                               padding=1, bias=True)
        self.weights_init()

    def make_conv_relu(self, feature_num_in, feature_num_out, is_downsampling=False):
        layers = []
        if is_downsampling:
            layers.append(nn.Conv2d(feature_num_in, feature_num_out, kernel_size=3, stride=2,
                                    padding=1, bias=True))
        else:
            layers.append(nn.Conv2d(feature_num_in, feature_num_out, kernel_size=3, stride=1,
                                    padding=1, bias=True))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def make_deconv_relu(self, feature_num_in, feature_num_out, is_upsampling=False):
        layers = []
        if is_upsampling:
            layers.append(nn.ConvTranspose2d(feature_num_in, feature_num_out, kernel_size=4, stride=2,
                                             padding=1, bias=True))
        else:
            layers.append(nn.ConvTranspose2d(feature_num_in, feature_num_out, kernel_size=3, stride=1,
                                             padding=1, bias=True))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        out_conv8 = self.conv8(out_conv7)
        out_conv9 = self.conv9(out_conv8)
        out_conv10 = self.conv10(out_conv9)

        out_deconv1 = self.deconv1(out_conv10)
        out_deconv2 = self.deconv2(out_deconv1)
        out_deconv3 = self.deconv3(out_deconv2 + out_conv8)
        out_deconv4 = self.deconv4(out_deconv3)
        out_deconv5 = self.deconv5(out_deconv4 + out_conv6)
        out_deconv6 = self.deconv6(out_deconv5)
        out_deconv7 = self.deconv7(out_deconv6 + out_conv4)
        out_deconv8 = self.deconv8(out_deconv7)
        out_deconv9 = self.deconv9(out_deconv8 + out_conv2)
        out_deconv10 = self.deconv10(out_deconv9)
        out = out_deconv10 + x
        output = torch.clamp(out, min=0, max=255)

        if target is not None:
            pairs = {'out': (out, target)}
            return pairs, self.exports(x, out, target)
        else:
            return self.exports(x, out, target)

    def exports(self, x, output, target):
        result = {'input': x, 'output': output}
        if target is not None:
            result['target'] = target
        return result

    def weights_init(self):
        for m in self.modules(): #deconv initial by pytorch original method(uniform, mean=0, variance=1/3*n)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.normal_(0, 0.0001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)


if __name__ == '__main__':
    gpu_id = 0
    red = Red20ResiDownsample(image_c=12).cuda(gpu_id)
    x = torch.autograd.Variable(torch.rand(4, 12, 64, 64).cuda(gpu_id))
    time_start = time.time()
    out = red(x)
    print "time consume: {} ms".format((time.time() - time_start) * 1000.0)
    print out['output'].size()
