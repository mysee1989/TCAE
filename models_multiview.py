import torch.nn as nn
from torch.autograd import Variable
import torch
import pdb

import numpy as np

class Au_detection(nn.Module):
	def __init__(self, input_fc=256, out_fc=12):
		super(Au_detection, self).__init__()
                self.batch_norm = nn.BatchNorm1d(input_fc)
                self.linear = nn.Linear(input_fc, out_fc, bias=False)

        def forward(self, in_put):
            out = self.batch_norm(in_put)
            out = self.linear(out)
            return out

class Gen_sep_feature(nn.Module):
    def __init__(self, output_size=256, num_filters = 32):
        super(Gen_sep_feature, self).__init__()
        # return pose feature, expression feature
        self.conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)  # 3 -> 32
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1) # 32 -> 64
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1) # 64 -> 128
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1) # 128 -> 256
        self.conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1) # 256 -> 256

        # each branch has separate 3 conv
        self.conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1) # 256 -> 256
        self.conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

        self.conv6_1 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1) # 256 -> 256
        self.conv7_1 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        self.conv8_1 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
        self.batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
        self.batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)

        self.batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

        self.batch_norm8_2_1 = nn.BatchNorm2d(num_filters * 8)
        self.batch_norm8_3_1 = nn.BatchNorm2d(num_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # general encoder
        x = self.leaky_relu(self.batch_norm(self.conv1(x)))
        x = self.leaky_relu(self.batch_norm2_0(self.conv2(x)))
        x = self.leaky_relu(self.batch_norm4_0(self.conv3(x)))
        x = self.leaky_relu(self.batch_norm8_0(self.conv4(x)))
        x = self.leaky_relu(self.batch_norm8_1(self.conv5(x)))

        pose_feature = self.leaky_relu(self.batch_norm8_2(self.conv6(x)))
        pose_feature = self.leaky_relu(self.batch_norm8_3(self.conv7(pose_feature)))
        pose_feature = self.conv8(pose_feature)

        emotion_feature = self.leaky_relu(self.batch_norm8_2_1(self.conv6_1(x)))
        emotion_feature = self.leaky_relu(self.batch_norm8_3_1(self.conv7_1(emotion_feature)))
        emotion_feature = self.conv8_1(emotion_feature)

        return pose_feature, emotion_feature


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=256, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))  # 4
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        return out_real.squeeze()



"""
generate 2 intermediate feature, 2 flow
"""
class FrontaliseModelMasks_wider(nn.Module):
	def __init__(self, num_decoders=5, inner_nc=128, num_additional_ids=0, \
                num_output_channels=2, smaller=False, num_masks=0):
		super(FrontaliseModelMasks_wider, self).__init__()
		print(num_additional_ids, inner_nc)

		# self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids)
                self.encoder = Gen_sep_feature()
		
		self.pose_decoder = self.generate_decoder_layers(inner_nc*2, num_output_channels=num_output_channels, num_filters=num_additional_ids)

		self.expression_decoder = self.generate_decoder_layers(inner_nc*2, num_output_channels=num_output_channels, num_filters=num_additional_ids)
                # pdb.set_trace()

                if num_masks > 0:
                    self.mask = self.generate_decoder_layers(inner_nc*2, num_output_channels=1, num_filters=num_additional_ids)


	def generate_encoder_layers(self, output_size=128, num_filters=64):
		pre_batch_norm = nn.BatchNorm2d(3)
		conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)  # 3 -> 32
		conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1) # 32 -> 64
		conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1) # 64 -> 128
		conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1) # 128 -> 256
		conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1) # 256 -> 256

                # each branch has separate 3 conv
		conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1) # 256 -> 256
		conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
		conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

		batch_norm = nn.BatchNorm2d(num_filters)
		batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
		batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
		batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

		leaky_relu = nn.LeakyReLU(0.2, True)
		return nn.Sequential(
                        pre_batch_norm, conv1, batch_norm, leaky_relu, 
                        conv2, batch_norm2_0, leaky_relu, 
                        conv3, batch_norm4_0, leaky_relu,
                        conv4, batch_norm8_0, leaky_relu, 
                        conv5, batch_norm8_1, leaky_relu, 
                        conv6, batch_norm8_2, leaky_relu, 
                        conv7, batch_norm8_3, leaky_relu, conv8)

	def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=32):
		up = nn.Upsample(scale_factor=2, mode='bilinear')

		dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
		dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
		dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
		dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
		dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
		dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
		dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
		dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

		batch_norm = nn.BatchNorm2d(num_filters)
		batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
		batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
		batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
		batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
		batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

		leaky_relu = nn.LeakyReLU(0.2)
		relu = nn.ReLU()
		tanh = nn.Tanh()

		return nn.Sequential(relu, 
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, relu,
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu, 
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, 
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv4, batch_norm8_7, relu, 
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, relu, 
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, 
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm, relu, 
                        nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
		

