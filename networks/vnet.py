import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlock_k(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, kernel_size = 3, normalization='none'):
        super(ConvBlock_k, self).__init__()
        self.n_stages = n_stages
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.kernel_size = kernel_size
        self.normalization = normalization
        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size, padding= (kernel_size-1)//2))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

    def config(self):     
        return {
            'name': ConvBlock_k.__name__,
            'n_stages': self.n_stages,
            'n_filters_in': self.n_filters_in,
            'n_filters_out': self.n_filters_out,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            # 'n_stages, n_filters_in, n_filters_out, kernel_size = 3, normalization='none''
            # **super(double_conv, self).__init__(),
        }

    @staticmethod
    def build_from_config(config):
        return ConvBlock_k(**config)


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):   
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.stride = stride
        self.normalization = normalization

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)


        return x
    
    def config(self):     
        return {
            'name': UpsamplingDeconvBlock.__name__,
            'n_filters_in': self.n_filters_in,
            'n_filters_out': self.n_filters_out,
            'stride': self.stride,
            'normalization': self.normalization,
            # n_filters_in, n_filters_out, stride=2, normalization='none'
            # **super(double_conv, self).__init__(),
        }

    @staticmethod
    def build_from_config(config):
        return UpsamplingDeconvBlock(**config)

class UpsamplingDeconvBlockd(nn.Module):  
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', interp_mode = 'trilinear'):
        super(UpsamplingDeconvBlockd, self).__init__()
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.stride = stride
        self.normalization = normalization
        self.interp_mode = interp_mode

        ops = []
        if normalization != 'none':
            # ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1, padding=0))
            if interp_mode == 'trilinear':
                ops.append(nn.Upsample(scale_factor=2, mode = interp_mode, align_corners = True))
            if interp_mode != 'trilinear':
                ops.append(nn.Upsample(scale_factor=2, mode = interp_mode))
            # ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1, padding=0))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1, padding=0))
            ops.append(nn.Upsample(scale_factor=2, mode = interp_mode))
            # ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1, padding=0))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def config(self):     
        return {
            'name': UpsamplingDeconvBlockd.__name__,
            'n_filters_in': self.n_filters_in,
            'n_filters_out': self.n_filters_out,
            'stride': self.stride,
            'normalization': self.normalization,
            'interp_mode': self.interp_mode,
            # n_filters_in, n_filters_out, stride=2, normalization='none', interp_mode = 'trilinear'
            # **super(double_conv, self).__init__(),
        }

    @staticmethod
    def build_from_config(config):
        return UpsamplingDeconvBlockd(**config)

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out


class VNet_withf(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_withf, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out, x9


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out, x9 = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, x9

class VNet_mct(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_mct, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)  
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_five_trilinear = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization) 
        self.block_five_up_trilinear = UpsamplingDeconvBlockd(n_filters * 16, n_filters * 8, normalization=normalization, interp_mode = 'trilinear')
        self.block_five_nearest = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)  
        self.block_five_up_nearest = UpsamplingDeconvBlockd(n_filters * 16, n_filters * 8, normalization=normalization, interp_mode = 'nearest')
        

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization) 
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_six_trilinear = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization) 
        self.block_six_up_trilinear = UpsamplingDeconvBlockd(n_filters * 8, n_filters * 4, normalization=normalization, interp_mode = 'trilinear')
        self.block_six_nearest = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization) 
        self.block_six_up_nearest = UpsamplingDeconvBlockd(n_filters * 8, n_filters * 4, normalization=normalization, interp_mode = 'nearest')

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization) 
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_seven_trilinear = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization) 
        self.block_seven_up_trilinear = UpsamplingDeconvBlockd(n_filters * 4, n_filters * 2, normalization=normalization, interp_mode = 'trilinear')
        self.block_seven_nearest = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization) 
        self.block_seven_up_nearest = UpsamplingDeconvBlockd(n_filters * 4, n_filters * 2, normalization=normalization, interp_mode = 'nearest')

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization) 
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
        self.block_eight_trilinear = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization) 
        self.block_eight_up_trilinear = UpsamplingDeconvBlockd(n_filters * 2, n_filters, normalization=normalization, interp_mode = 'trilinear')
        self.block_eight_nearest = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization) 
        self.block_eight_up_nearest = UpsamplingDeconvBlockd(n_filters * 2, n_filters, normalization=normalization, interp_mode = 'nearest')

        
        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        
        self.block_nine_trilinear = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv_trilinear = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        
        self.block_nine_nearest = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv_nearest = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)

        
        x5_up_trilinear = self.block_five_up_trilinear(x5)
        x5_up_trilinear = x5_up_trilinear + x4

        x6_trilinear = self.block_six_trilinear(x5_up_trilinear)
        x6_up_trilinear = self.block_six_up_trilinear(x6_trilinear)
        x6_up_trilinear = x6_up_trilinear + x3

        x7_trilinear = self.block_seven_trilinear(x6_up_trilinear)
        x7_up_trilinear = self.block_seven_up_trilinear(x7_trilinear)
        x7_up_trilinear = x7_up_trilinear + x2

        x8_trilinear = self.block_eight_trilinear(x7_up_trilinear)
        x8_up_trilinear = self.block_eight_up_trilinear(x8_trilinear)
        x8_up_trilinear = x8_up_trilinear + x1
        x9_trilinear = self.block_nine_trilinear(x8_up_trilinear)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9_trilinear = self.dropout(x9_trilinear)
        out_trilinear = self.out_conv_trilinear(x9_trilinear)

        
        x5_up_nearest = self.block_five_up_nearest(x5)
        x5_up_nearest = x5_up_nearest + x4

        x6_nearest = self.block_six_nearest(x5_up_nearest)
        x6_up_nearest = self.block_six_up_nearest(x6_nearest)
        x6_up_nearest = x6_up_nearest + x3

        x7_nearest = self.block_seven_nearest(x6_up_nearest)
        x7_up_nearest = self.block_seven_up_nearest(x7_nearest)
        x7_up_nearest = x7_up_nearest + x2

        x8_nearest = self.block_eight_nearest(x7_up_nearest)
        x8_up_nearest = self.block_eight_up_nearest(x8_nearest)
        x8_up_nearest = x8_up_nearest + x1
        x9_nearest = self.block_nine_nearest(x8_up_nearest)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9_nearest = self.dropout(x9_nearest)
        out_nearest = self.out_conv_nearest(x9_nearest)
        return out, out_trilinear, out_nearest


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out1, out2, out3 = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out1, out2, out3

