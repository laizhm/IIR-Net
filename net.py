import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import kornia
import os
import src.params as params

PRETAINED_WEIGHT_PATH = 'networks/xception-b5690688.pth'
pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975  # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}

class Two_Stream_Net(nn.Module):
    def __init__(self, xcep=True):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=6, return_fea=True, xcep=xcep)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True, xcep=xcep)
        self.xception_reshape = TransferModel(
            'xception', dropout=0.5, inc=9, return_fea=True, xcep=xcep)
        self.xception_PLGF = TransferModel(
            'xception', dropout=0.5, inc=27, return_fea=True, xcep=xcep)
        self.PLGF_descriptor = LearnablePLGF(kernel_size=3, out=out)
        
        self.decom1 = DecomNet(256)
        self.decom2 = DecomNet(256)
        
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)

        self.reshape_conv1 = SRMConv2d_Separate(32, 32)
        self.reshape_conv2 = SRMConv2d_Separate(64, 64)

        self.PLGF_conv1 = SRMConv2d_Separate(32, 32)
        self.PLGF_conv2 = SRMConv2d_Separate(64, 64)

        self.relu = nn.ReLU(inplace=True)

        self.srm_sa = SRMPixelAttention(3)
        self.reshape_sa = SRMPixelAttention(9, srm=False)
        self.PLGF_sa = SRMPixelAttention(27, srm=False)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)

        self.att1 = FeatureFusionModule(in_chan=32 * 2, out_chan=32)
        self.att2 = FeatureFusionModule(in_chan=32 * 2, out_chan=32)
        self.att13 = FeatureFusionModule(in_chan=32 * 2, out_chan=32)

        self.att3 = FeatureFusionModule(in_chan=64 * 2, out_chan=64)
        self.att4 = FeatureFusionModule(in_chan=64 * 2, out_chan=64)
        self.att14 = FeatureFusionModule(in_chan=64 * 2, out_chan=64)

        self.att5 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
        self.att6 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
        self.att15 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)

        self.att7 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
        self.att8 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
        self.att16 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)

        self.att9 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
        self.att10 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
        self.att17 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)

        self.att11 = FeatureFusionModule(in_chan=2048 * 2, out_chan=2048)
        self.att12 = FeatureFusionModule(in_chan=2048 * 2, out_chan=2048)
        self.att18 = FeatureFusionModule(in_chan=2048 * 2, out_chan=2048)


        self.fusion = FeatureFusionModule()
        self.end_linear = AngleSimpleLinear(2048, 2)
        self.seg_decoder = SegmentationDecoder(2048)

    def reshape(self, x, x_next):
        illu, ref, ref_2 = self.decom1(x/255)
        illu2, ref2, ref2_2 = self.decom2(x_next/255)

        input_RGB_RE = illu2 + ref
        input_RGB_RE2 = illu + ref2

        input_RGB_RE_minus = input_RGB_RE-input_RGB_RE2

        input_RGB_RE = torch.exp(illu2 + ref)

        input_RGB_RE_minus = torch.exp(input_RGB_RE_minus)

        out = torch.cat([x, input_RGB_RE, input_RGB_RE_minus], dim=1)
        return out

    def features(self, x, x_next):
        srm = self.srm_conv0(x/255)
        yuv = kornia.color.rgb_to_yuv(x/255)
        yuv[:, 1:3, :, :] += 0.5

        reshape = self.reshape(x, x_next)
        plgf = self.PLGF_descriptor(x)/255

        reshape_0 = self.xception_reshape.model.fea_part1_0(reshape)
        plgf_0 = self.xception_PLGF.model.fea_part1_0(plgf)

        x = self.xception_rgb.model.fea_part1_0(torch.cat([x/255, yuv], dim=1))
        y_srm = self.xception_srm.model.fea_part1_0(srm) + self.srm_conv1(x)
        y_reshape =  reshape_0 + self.reshape_conv1(x)

        y_plgf =  plgf_0 + self.PLGF_conv1(x)
        y_plgf = self.relu(y_plgf)

        y = self.att1(y_srm, y_reshape)
        y = self.att2(y, y_plgf)
        y = self.relu(y)


        x = self.xception_rgb.model.fea_part1_1(x)
        y_srm = self.xception_srm.model.fea_part1_1(y) + self.srm_conv2(x)
        y_reshape = self.xception_reshape.model.fea_part1_1(y_reshape) + self.reshape_conv2(x)
        y_plgf = self.xception_PLGF.model.fea_part1_1(y_plgf) + self.PLGF_conv2(x)
        
        y = self.att3(y_srm, y_reshape)
        y = self.att4(y, y_plgf)
        y = self.relu(y)


        x = x * self.srm_sa(srm) +  x * self.reshape_sa(reshape) + x * self.PLGF_sa(plgf) + x
        x = self.srm_sa_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        y_srm = self.xception_srm.model.fea_part2(y)
        y_reshape = self.xception_reshape.model.fea_part2(y_reshape)
        y_plgf = self.xception_PLGF.model.fea_part2(y_plgf)
        y = self.att5(y_srm, y_reshape)
        y = self.att6(y, y_plgf)
        y = self.relu(y)

        x, y = self.dual_cma0(x, y)


        x = self.xception_rgb.model.fea_part3(x)        
        y_srm = self.xception_srm.model.fea_part3(y)
        y_reshape = self.xception_reshape.model.fea_part3(y_reshape)
        y_plgf = self.xception_PLGF.model.fea_part3(y_plgf)
        y = self.att7(y_srm, y_reshape)
        y = self.att8(y, y_plgf)
        y = self.relu(y)

        x, y = self.dual_cma1(x, y)

        x = self.xception_rgb.model.fea_part4(x)
        y_srm = self.xception_srm.model.fea_part4(y)
        y_reshape = self.xception_reshape.model.fea_part4(y_reshape)
        y_plgf = self.xception_PLGF.model.fea_part4(y_plgf)
        y = self.att9(y_srm, y_reshape)
        y = self.att10(y, y_plgf)
        y = self.relu(y)
    
        x = self.xception_rgb.model.fea_part5(x)
        y_srm = self.xception_srm.model.fea_part5(y)
        y_reshape = self.xception_reshape.model.fea_part5(y_reshape)
        y_plgf = self.xception_PLGF.model.fea_part5(y_plgf)
        y = self.att11(y_srm, y_reshape)
        y = self.att12(y, y_plgf)

        fea = self.fusion(x, y)

        return fea

    def features_aug(self, x):
        srm = self.srm_conv0(x/255)
        yuv = kornia.color.rgb_to_yuv(x/255)
        yuv[:, 1:3, :, :] += 0.5

        plgf = self.PLGF_descriptor(x)/255

        plgf_0 = self.xception_PLGF.model.fea_part1_0(plgf)

        x = self.xception_rgb.model.fea_part1_0(torch.cat([x/255, yuv], dim=1))
        y_srm = self.xception_srm.model.fea_part1_0(srm) \
            + self.srm_conv1(x)

        y_plgf =  plgf_0 + self.PLGF_conv1(x)
        y_plgf = self.relu(y_plgf)

        y = self.att13(y_srm, y_plgf)
        y = self.relu(y)


        x = self.xception_rgb.model.fea_part1_1(x)
        y_srm = self.xception_srm.model.fea_part1_1(y) \
            + self.srm_conv2(x)

        y_plgf = self.xception_PLGF.model.fea_part1_1(y_plgf) \
            + self.PLGF_conv2(x)
        
        y_plgf = self.relu(y_plgf)

        y = self.att14(y_srm, y_plgf)
        y = self.relu(y)


        x = x * self.srm_sa(srm) + x * self.PLGF_sa(plgf) + x
        x = self.srm_sa_post(x)

        x = self.xception_rgb.model.fea_part2(x)
        y_srm = self.xception_srm.model.fea_part2(y)
        y_plgf = self.xception_PLGF.model.fea_part2(y_plgf)
        y = self.att15(y_srm, y_plgf)
        y = self.relu(y)

        x, y = self.dual_cma0(x, y)


        x = self.xception_rgb.model.fea_part3(x)        
        y_srm = self.xception_srm.model.fea_part3(y)
        y_plgf = self.xception_PLGF.model.fea_part3(y_plgf)
        y = self.att16(y_srm, y_plgf)
        y = self.relu(y)

        x, y = self.dual_cma1(x, y)

        x = self.xception_rgb.model.fea_part4(x)
        y_srm = self.xception_srm.model.fea_part4(y)
        y_plgf = self.xception_PLGF.model.fea_part4(y_plgf)
        y = self.att17(y_srm, y_plgf)
        y = self.relu(y)
    
        x = self.xception_rgb.model.fea_part5(x)
        y_srm = self.xception_srm.model.fea_part5(y)
        y_plgf = self.xception_PLGF.model.fea_part5(y_plgf)
        y = self.att18(y_srm, y_plgf)

        fea = self.fusion(x, y)
                

        return fea


    def classifier(self, fea):
        out, fea_pool, fea = self.xception_rgb.classifier(fea)
        return out, fea_pool, fea

    def forward(self, input_RGB_origin, input_RGB_aug, input_RGB_origin_next):
        out, fea_pool, fea = self.classifier(self.features(input_RGB_origin, input_RGB_origin_next))
        out = self.end_linear(fea_pool)
        mask_out = self.seg_decoder(fea)

        out2, fea_pool2, fea2 = self.classifier(self.features_aug(input_RGB_aug))
        out2 = self.end_linear(fea_pool2)
        mask_out2 = self.seg_decoder(fea2)
        return out, mask_out, out2, mask_out2, fea_pool, fea
        
class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """

    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0,
                 weight_norm=False, return_fea=False, inc=3, xcep=True):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        self.return_fea = return_fea

        if modelchoice == 'xception':

            def return_pytorch04_xception(pretrained=True):
                # Raises warning "src not broadcastable to dst" but thats fine
                model = xception(pretrained=False)
                if pretrained:
                    # Load model in torch 0.4+
                    model.fc = model.last_linear
                    del model.last_linear
                    state_dict = torch.load(
                        PRETAINED_WEIGHT_PATH)
                    for name, weights in state_dict.items():
                        if 'pointwise' in name:
                            state_dict[name] = weights.unsqueeze(
                                -1).unsqueeze(-1)
                    model.load_state_dict(state_dict)
                    model.last_linear = model.fc
                    del model.fc
                return model

            self.model = return_pytorch04_xception(xcep)
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                if weight_norm:
                    print('Using Weight_Norm')
                    self.model.last_linear = nn.utils.weight_norm(
                        nn.Linear(num_ftrs, num_out_classes), name='weight')
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                if weight_norm:
                    print('Using Weight_Norm')
                    self.model.last_linear = nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.utils.weight_norm(
                            nn.Linear(num_ftrs, num_out_classes), name='weight')
                    )

                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )

            if inc != 3:
                self.model.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
                nn.init.xavier_normal(self.model.conv1.weight.data, gain=0.02)

        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean=False, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on lib, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise NotImplementedError('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        out, x = self.model(x)
        if self.return_fea:
            return out, x
        else:
            return out

    def features(self, x):
        x = self.model.features(x)
        return x

    def classifier(self, x):
        out, x_pool, x = self.model.classifier(x)
        return out, x_pool, x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, inc=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        #TODO last_linear
        self.fc = nn.Linear(2048, num_classes)
        # self.fc = AngleSimpleLinear(2048, 2)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------
    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x

    def fea_part4(self, x):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        return x

    def fea_part5(self, x):
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def features(self, input):
        x = self.fea_part1(input)

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)
        return x

    def classifier(self, features):
        x = self.relu(features) 
        x_pool = F.adaptive_avg_pool2d(x, (1, 1))
        x_pool = x_pool.view(x.size(0), -1)
        out = self.last_linear(x_pool)
        return out, x_pool, x

    def forward(self, input):
        x = self.features(input)
        out, x_pool, x = self.classifier(x)
        return out, x_pool

def xception(num_classes=1000, pretrained='imagenet', inc=3):
    model = Xception(num_classes=num_classes, inc=inc)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(
                settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model
class GRL(nn.Module):
    """梯度反转层"""
    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        """
        coeff=[2/(1+exp(-alpha*p))]-1，其中alpha=10，p随着训练进行由0变为1，所以coeff随着训练也会由0变为1。
        当p>1时不影响，因为此时exp(-alpha*p)会更接近0
        （1）这表明训练开始时，域分类损失不会反向传播到编码器网络中，只有域分类器得到训练；而随着训练进行，编码器得到训练，并开始逐步生成可以混淆领域分类器的特征。
        （2）即需要先训练具有分类能力的编码器网络，然后通过对抗学习得到域不变特征。如果太早进行对抗学习，此时编码器网络还不具备较好的分类能力
        """
        p = self.iter_num / self.max_iter
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * p))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput


class Discriminator(nn.Module):
    """域判别器，用于提取域不变的内容特征"""
    def __init__(self, input_dim, domain_num, max_iter):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim                      # 输入特征的维度
        self.domain_num = domain_num                    # 训练数据的域数量
        self.fc11 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, self.domain_num)        # 这里domain_num表示训练集中的域数据
        self.ad_net = nn.Sequential(
            self.fc11,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2,
            nn.Softmax(dim=1),
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature)) * 4
        return adversarial_out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        # return F.interpolate(x, size=(H,W), mode='bilinear') + y #
        return F.interpolate(x, size=(H, W), mode='nearest') + y  #

    def _downsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = x.size()
        # return F.interpolate(x, size=(H,W), mode='bilinear') + y #
        return F.interpolate(y, size=(H, W), mode='nearest') + x  #

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.latlayer1(c5)

        p4 = self._upsample_add(p5, self.latlayer2(c4))

        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p7


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


def FPN101():
    return FPN(Bottleneck, [2, 4, 23, 3])


import torch
import torch.nn as nn
import torch.fft


def get_srm_kernel() -> torch.Tensor:
    """返回经典 5x5 SRM 核 (1,1,5,5) 已归一化"""
    kernel = torch.tensor([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=torch.float32) / 12.0
    return kernel.view(1, 1, 5, 5)


class FreqDomainSRM(nn.Module):
    """
    超高效 FFT 实现的 SRM 残差提取（全局卷积）
    优势：
    - 使用 rfft2/irfft2 → 内存和速度 ≈ fft2 的 50%
    - 自动支持任意奇/偶尺寸（256、257、512 等都完美）
    - 数值更稳定（实部直接输出，无需手动 .real）
    - 代码极简，易于维护
    """

    def __init__(self, image_size=(256, 256), trunc_val=2.0):
        super().__init__()
        H, W = image_size

        # 1. 构造中心填充的空域滤波器
        kernel = get_srm_kernel()  # (1,1,5,5)
        padded = torch.zeros(1, 1, H, W, dtype=torch.float32)
        kh, kw = 5, 5
        pad_h = (H - kh) // 2
        pad_w = (W - kw) // 2
        padded[:, :, pad_h:pad_h + kh, pad_w:pad_w + kw] = kernel

        # 2. fftshift：把零频移到左上角（PyTorch FFT 要求）
        padded = torch.fft.fftshift(padded, dim=(-2, -3))

        # 3. 实数 FFT（rfft2）→ 只需要一半频谱，极大节省显存和计算量
        filter_rfreq = torch.fft.rfft2(padded)  # (1,1,H,W//2+1) complex

        # 注册 buffer（支持 .cuda()/.cpu() 自动迁移）
        self.register_buffer("filter_rfreq_real", filter_rfreq.real)
        self.register_buffer("filter_rfreq_imag", filter_rfreq.imag)

        self.H = H
        self.W = W
        self.trunc_val = trunc_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) float32，建议值域 [0,1] 或 [-1,1]
        返回: 同形残差图 (B, C, H, W)
        """
        if x.shape[-2:] != (self.H, self.W):
            raise ValueError(f"Input size must be {self.H}x{self.W}, got {x.shape[-2:]}")

        # 重建复数频域滤波器
        filter_rfreq = torch.complex(self.filter_rfreq_real, self.filter_rfreq_imag)

        # 实数 FFT + 频域相乘 + 实数 IFFT（最快最省）
        x_rfreq = torch.fft.rfft2(x)  # (B,C,H,W//2+1)
        filtered = x_rfreq * filter_rfreq  # 自动广播
        residual = torch.fft.irfft2(filtered, s=(self.H, self.W))  # 直接实数输出
        return torch.clamp(residual, -self.trunc_val, self.trunc_val)


class Illumination_Net(nn.Module):
    """
    基于光照信息的换脸检测网络
    2024.07.30，ljc
    """
    def __init__(self, num_class=2, num_domains=4, max_iter=4000):
        super(Illumination_Net, self).__init__()

        # 定义可学习的PLGF层
        # self.srm = FreqDomainSRM(image_size=(256, 256))
        self.PLGF_descriptor = LearnablePLGF(kernel_size=3)
        # self.PLGF_descriptor = OriPLGF(kernel_size=3)

        # 加载骨干网络，todo:resnet18
        model_resnet = models.resnet18(pretrained=True)#todo gzl修改 resnet18
        model_resnet2 = models.resnet18(pretrained=True)  # todo gzl修改 resnet18
        print("Backbone Network: Resnet18\n")# todo gzl
        # f_acc_valid = open(params.result_valid_file, "a+")
        # f_acc_valid.write(("Backbone Network: ResNet34\n"))
        # f_acc_valid.close()
        # Branch1: Illumination-related，使用BN层（适用于认知任务，如分类、分割等）
        self.IRE_input_layer = nn.Sequential(
            model_resnet.conv1,     # 输入通道为3，通道数64,ks=7,stride=2,p=3,sz=128
            model_resnet.bn1,
            model_resnet.relu,
            model_resnet.maxpool    # 通道数64,ks=3,stride=2,p=1,sz=64
        )

        self.IRE_layer1 = model_resnet.layer1  # 通道数64,ks=3,stride=1,sz=64

        self.IRE_layer2 = model_resnet.layer2  # 通道数128,ks=3,stride=2,sz=32
        self.IRE_layer3 = model_resnet.layer3  # 通道数256,ks=3,stride=2,sz=16
        self.IRE_layer4 = model_resnet.layer4  # 通道数512,ks=3,stride=2,sz=8
        self.IRE_output_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),#gzl todo 512 512
            nn.BatchNorm2d(512),#todo 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),#todo 512 512
            nn.BatchNorm2d(512),# todo 512
            nn.ReLU(inplace=True)
        )  # 通道数512,ks=3,stride=1,sz=8

        # Branch2: Illumination-invariant，使用BN层（适用于认知任务，如分类、分割等）
        self.IIE_input_layer = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # model_resnet2.conv1,  # 输入通道为3，通道数64,ks=7,stride=2,p=3,sz=128
            model_resnet2.bn1,
            model_resnet2.relu,
            model_resnet2.maxpool  # 通道数64,ks=3,stride=2,p=1,sz=64
        )
        self.IIE_layer1 = model_resnet2.layer1  # 通道数64,ks=3,stride=1,sz=64
        self.IIE_layer2 = model_resnet2.layer2  # 通道数128,ks=3,stride=2,sz=32
        self.IIE_layer3 = model_resnet2.layer3  # 通道数256,ks=3,stride=2,sz=16
        self.IIE_layer4 = model_resnet2.layer4  # 通道数512,ks=3,stride=2,sz=8
        self.IIE_output_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),# todo gzl 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 通道数512,ks=3,stride=1,sz=8

        # Lateral layers
        self.IRE_latlayer1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)#todo gzl 64 128
        self.IRE_latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)#todo gzl 128 256
        self.IRE_latlayer3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)#todo gzl 256 512

        self.IIE_latlayer1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)# todo 同上
        self.IIE_latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)# todo 同上
        self.IIE_latlayer3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)# todo 同上

        # Down-top layers
        # self.downlayer1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)#todo 128 256
        # self.downlayer2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)# todo 256 512
        # self.downlayer3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)# todo 512 512
        self.downlayer1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)#todo 128 256
        self.downlayer2 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)# todo 256 512
        self.downlayer3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)# todo 512 512

        self.mfm_downlayer1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.mfm_downlayer2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)

        # classifier
        self.classifier = nn.Linear(4096, num_class, bias=True)
        # self.classifier = nn.Linear(512, num_class, bias=True)
        self.classifier2 = nn.Linear(512, num_class, bias=True)
        self.classifier3 = nn.Linear(512, num_class, bias=True)

        # 域判别器，例如使用FF++的4个子集联合训练，则DID为0~4（0为真，1~4分别为四个伪造数据集）
        self.discriminator = Discriminator(input_dim=512, domain_num=num_domains, max_iter=max_iter)

        self.separate_module = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=4096,
        #         out_channels=2048,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(
        #         in_channels=2048,
        #         out_channels=1024,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(
        #         in_channels=1024,
        #         out_channels=512,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(
        #         in_channels=512,
        #         out_channels=256,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(
        #         in_channels=256,
        #         out_channels=2,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(inplace=True)
        # )
        self.fea_att = nn.Parameter(
            torch.ones([1, 4096, 1, 1], dtype=torch.float32),
            requires_grad=True
        )


    def getFeatures(self, input_RGB):
        """输入RGB图像，得到两个支路的输出特征，以及融合特征"""
        # print(input_RGB.max())
        # print(input_RGB.shape)
        # print(type(input_RGB))
        # mean = torch.tensor([0.485, 0.456, 0.406], device=input_RGB.device).view(-1, 1, 1).unsqueeze(0)
        # std = torch.tensor([0.229, 0.224, 0.225], device=input_RGB.device).view(-1, 1, 1).unsqueeze(0)
        # input_RGB_norm = (input_RGB - mean) / std
        # input_RGB = input_RGB.norm()

        # print(input_RGB_norm.max())

        # 计算支路1的特征
        # print("self.IRE_input_layer(input_RGB)", input_RGB.max())
        fea_IRE_0 = self.IRE_input_layer(input_RGB / 255)                                 # [N,64,64,64]
        # print("self.IRE_layer1(fea_IRE_0)", fea_IRE_0.shape)
        fea_IRE_1 = self.IRE_layer1(fea_IRE_0)                                      # [N,64,64,64]
        # print("self.IRE_layer2(fea_IRE_1)", fea_IRE_1.shape)
        fea_IRE_2 = self.IRE_layer2(fea_IRE_1)
        # print("self.IRE_layer2(fea_IRE_1)", fea_IRE_2.shape)
        fea_IRE_1_mid = self.IRE_latlayer1(fea_IRE_1)# [N,128,32,32]
        # print("self.IRE_latlayer1(fea_IRE_1)", fea_IRE_1_mid.shape)
        # print("self._downsample_add(fea_IRE_2, fea_IRE_1_mid)", fea_IRE_2.shape)
        # fea_IRE_2 = self._downsample_add(fea_IRE_2, self.IRE_latlayer1(fea_IRE_1))
        fea_IRE_2 = self._downsample_add(fea_IRE_2, fea_IRE_1_mid)  # [N,128,32,32]
        # print("self._downsample_add(fea_IRE_2, fea_IRE_1_mid)", fea_IRE_2.shape)
        fea_IRE_3 = self.IRE_layer3(fea_IRE_2)                                      # [N,256,16,16]
        # print("self.IRE_layer3(fea_IRE_2)", fea_IRE_3.max())
        # fea_IRE_3 = self._downsample_add(fea_IRE_3, self.IRE_latlayer2(fea_IRE_2))  # [N,256,16,16]
        fea_IRE_2_mid = self.IRE_latlayer2(fea_IRE_2)
        # print("self.IRE_latlayer2(fea_IRE_2)", fea_IRE_2_mid.max())
        fea_IRE_3 = self._downsample_add(fea_IRE_3, fea_IRE_2_mid)
        # print("self._downsample_add(fea_IRE_3, fea_IRE_2_mid)", fea_IRE_3.max())
        fea_IRE_4 = self.IRE_layer4(fea_IRE_3)                                      # [N,512,8,8]
        # print("self.IRE_layer4(fea_IRE_3)", fea_IRE_4.max())
        # fea_IRE_4 = self._downsample_add(fea_IRE_4, self.IRE_latlayer3(fea_IRE_3))  # [N,512,8,8]
        fea_IRE_3_mid = self.IRE_latlayer3(fea_IRE_3)
        # print("self.IRE_latlayer3(fea_IRE_3)", fea_IRE_3_mid.max())
        fea_IRE_4 = self._downsample_add(fea_IRE_4, fea_IRE_3_mid)
        # print("self._downsample_add(fea_IRE_4, fea_IRE_3_mid)", fea_IRE_4.max())
        fea_IRE_5 = self.IRE_output_layer(fea_IRE_4)                                # [N,512,8,8]
        # print("self.IRE_output_layer(fea_IRE_4)", fea_IRE_5.max())
        # 计算支路2的特征
        input_PLGF = self.PLGF_descriptor(input_RGB)
        # input_PLGF = self.srm(input_RGB)
        # print("self.PLGF_descriptor(input_RGB)", input_PLGF.max())
        # import cv2, os
        # from PIL import Image
        # for i in range(input_PLGF.shape[0]):
        #     # mean = torch.tensor([0.485, 0.456, 0.406], device=input_RGB.device).view(-1, 1, 1).unsqueeze(0)
        #     # std = torch.tensor([0.229, 0.224, 0.225], device=input_RGB.device).view(-1, 1, 1).unsqueeze(0)
        #     # 执行反向标准化: x_unnorm = x_norm * std + mean
        #     # out = input_PLGF * std + mean
        #     # out = out.clamp(0, 1) * 255
        #     # ori = input_RGB * std + mean
        #     # ori = ori.clamp(0, 1) * 255
        #     out = input_PLGF * 255
        #     out = out.detach().permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #     ori = input_RGB.detach().permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :].astype(np.uint8)
        #     # cv2.imwrite(os.path.join("./data/test", str(i) + "_PLGF.jpg"), out)
        #     # cv2.imwrite(os.path.join("./data/test", str(i) + "_ori.jpg"), ori)
        #     Image.fromarray(out).save(os.path.join("./data/test", str(i) + "_PLGF.jpg"))
        #     Image.fromarray(ori).save(os.path.join("./data/test", str(i) + "_ori.jpg"))
        #     print(os.path.join("./data/test", str(i) + "_ori.jpg"))
        fea_IIE_0 = self.IIE_input_layer(input_PLGF)                                # [N,64,64,64]
        fea_IIE_1 = self.IIE_layer1(fea_IIE_0)                                      # [N,64,64,64]
        fea_IIE_2 = self.IIE_layer2(fea_IIE_1)
        fea_IIE_1_mid = self.IIE_latlayer1(fea_IIE_1)# [N,128,32,32]
        # fea_IIE_2 = self._downsample_add(fea_IIE_2, self.IIE_latlayer1(fea_IIE_1))  # [N,128,32,32]
        fea_IIE_2 = self._downsample_add(fea_IIE_2, fea_IIE_1_mid)  # [N,128,32,32]
        fea_IIE_3 = self.IIE_layer3(fea_IIE_2)
        fea_IIE_2_mid = self.IIE_latlayer2(fea_IIE_2)# [N,256,16,16]
        fea_IIE_3 = self._downsample_add(fea_IIE_3, fea_IIE_2_mid)
        # fea_IIE_3 = self._downsample_add(fea_IIE_3, self.IIE_latlayer2(fea_IIE_2))  # [N,256,16,16]
        fea_IIE_4 = self.IIE_layer4(fea_IIE_3)                                      # [N,512,8,8]
        fea_IIE_3_mid = self.IIE_latlayer3(fea_IIE_3)
        fea_IIE_4 = self._downsample_add(fea_IIE_4, fea_IIE_3_mid)
        # fea_IIE_4 = self._downsample_add(fea_IIE_4, self.IIE_latlayer3(fea_IIE_3))  # [N,512,8,8]
        fea_IIE_5 = self.IIE_output_layer(fea_IIE_4)                                # [N,512,8,8]

        # 计算中间融合层特征
        # fea_MFM_1 = (self.IRE_latlayer1(fea_IRE_1)+self.IIE_latlayer1(fea_IIE_1)) / 2       # [N,64,64,64]
        fea_MFM_1 = torch.cat([fea_IRE_1_mid, fea_IIE_1_mid], dim=1)
        # fea_MFM_2 = (self.IRE_latlayer2(fea_IRE_2)+self.IIE_latlayer2(fea_IIE_2)) / 2       # [N,128,32,32]
        fea_MFM_2 = torch.cat([fea_IRE_2_mid, fea_IIE_2_mid], dim=1)
        # fea_MFM_2 = self._downsample_add(fea_MFM_2, self.downlayer1(fea_MFM_1))             # [N,128,32,32]
        fea_MFM_2 = self.mfm_downlayer1(torch.cat([fea_MFM_2, self.downlayer1(fea_MFM_1)], dim=1))
        # fea_MFM_3 = (self.IRE_latlayer3(fea_IRE_3) + self.IIE_latlayer3(fea_IIE_3)) / 2     # [N,256,16,16]
        fea_MFM_3 = torch.cat([fea_IRE_3_mid, fea_IIE_3_mid], dim=1)
        # fea_MFM_3 = self._downsample_add(fea_MFM_3, self.downlayer2(fea_MFM_2))             # [N,512,16,16]
        fea_MFM_3 = self.mfm_downlayer2(torch.cat([fea_MFM_3, self.downlayer2(fea_MFM_2)], dim=1))
        fea_MFM_4 = self.downlayer3(fea_MFM_3)                                              # [N,512,8,8]

        fea_fusion = (torch.cat([fea_IRE_5, fea_IIE_5, fea_MFM_4], dim=1))

        separate_out = self.separate_module(fea_fusion)
        # fea_fusion = (fea_IRE_5 + fea_IIE_5 + fea_MFM_4) / 3

        fea_fusion = (torch.cat([fea_fusion, separate_out], dim=1) ) * self.fea_att.to(input_RGB.device)
        # img_separation = self.up(fea_fusion)
        return fea_fusion, fea_IRE_5, fea_IIE_5, separate_out, fea_fusion
        # return fea_fusion, fea_IRE_5, fea_IIE_5, separate_out, img_separation

    def forward(self, input_RGB_origin, input2_RGB_random):
        """
        input_RGB_origin和input2_RGB_random分别为原始域排列的输入和域随机化后的输入
        其中，input2_RGB_random是使用Relighting方法在input_RGB_origin上生成得到
        """
        # 分别提取两种域排序的特征
        fea_fusion_ori, fea_IRE_ori, fea_IIE_ori, separate_out, img_separation = self.getFeatures(input_RGB_origin)       # [N,512,8,8]
        fea_fusion_rand, fea_IRE_rand, fea_IIE_rand, separate_out, img_separation = self.getFeatures(input2_RGB_random)   # [N,512,8,8]

        # 使用fea_fusion_ori进行分类
        f_cls_mid = torch.nn.functional.adaptive_avg_pool2d(fea_fusion_ori, 1)                  # out=[B,512,1,1]
        # print("torch.nn.functional.adaptive_avg_pool2d(fea_fusion_ori, 1)", f_cls_mid.max())
        f_cls = f_cls_mid.reshape(f_cls_mid.shape[0], -1)                                           # out=[B,512]
        # print("f_cls_mid.reshape(f_cls_mid.shape[0], -1)", f_cls.max())
        pred_cls = self.classifier(f_cls)
        pred_cls = nn.Softmax(dim=1)(pred_cls)
        # print("self.classifier(f_cls)", pred_cls.max())
        # 使用fea_fusion_rand进行分类
        f_cls2 = torch.nn.functional.adaptive_avg_pool2d(fea_fusion_rand, 1)  # out=[B,512,1,1]
        f_cls2 = f_cls2.reshape(f_cls2.shape[0], -1)  # out=[B,512]
        pred_cls2 = self.classifier(f_cls2)
        pred_cls2 = nn.Softmax(dim=1)(pred_cls2)
        # out=[B,num_class]

        # 将原始和生成的光照不变特征拼接，更全面
        # f_CR_pairs = torch.nn.functional.adaptive_avg_pool2d(fea_IIE_ori, 1)    # out=[B,512,1,1]

        f_CR_pairs_mid = torch.nn.functional.adaptive_avg_pool2d(fea_IRE_ori, 1)  # out=[B,512,1,1]
        # print("torch.nn.functional.adaptive_avg_pool2d(fea_IRE_ori, 1)", f_CR_pairs_mid.max())
        f_CR_pairs = f_CR_pairs_mid.reshape(f_CR_pairs_mid.shape[0], -1)                # out=[B,512]
        # print("f_CR_pairs_mid.reshape(f_CR_pairs_mid.shape[0], -1)", f_CR_pairs.max())

        # # 使用fea_IIE_ori进行分类
        # f_IIE_mid = torch.nn.functional.adaptive_avg_pool2d(fea_IIE_ori, 1)  # out=[B,512,1,1]
        # f_IIE = f_IIE_mid.reshape(f_IIE_mid.shape[0], -1)  # out=[B,512]
        # pred_IIE = self.classifier2(f_IIE)
        # # 使用fea_IIE_rand进行分类
        # f_IIE2 = torch.nn.functional.adaptive_avg_pool2d(fea_IIE_rand, 1)  # out=[B,512,1,1]
        # f_IIE2 = f_IIE2.reshape(f_IIE2.shape[0], -1)  # out=[B,512]
        # pred_IIE2 = self.classifier2(f_IIE2)

        # # 使用fea_IRE_ori进行分类
        # pred_IRE = self.classifier3(f_CR_pairs)
        # # 使用fea_IRE_rand进行分类
        # f_IRE2 = torch.nn.functional.adaptive_avg_pool2d(fea_IRE_rand, 1)  # out=[B,512,1,1]
        # f_IRE2 = f_IRE2.reshape(f_IRE2.shape[0], -1)  # out=[B,512]
        # pred_IRE2 = self.classifier3(f_IRE2)

        # 此时对应的域标签应该选择Dt的，即使用谁的内容，特征就是哪个域的
        pred_dis_invariant = self.discriminator(f_CR_pairs)                     # out=[B,num_domains]
        # print("self.discriminator(f_CR_pairs)", pred_dis_invariant.max())
        return pred_cls, pred_cls2, fea_IRE_ori, fea_IRE_rand, fea_IIE_ori, fea_IIE_rand,  pred_dis_invariant, separate_out, img_separation

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        # return F.interpolate(x, size=(H,W), mode='bilinear') + y #
        return F.interpolate(x, size=(H, W), mode='nearest') + y  #

    def _downsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''

        _, _, H, W = x.size()
        # # return F.interpolate(x, size=(H,W), mode='bilinear') + y #
        return F.interpolate(y, size=(H, W), mode='nearest') + x
#         ##############todo gzl###########
#         # 获取目标尺寸
#         _, _, H, W = x.size()
#
#         # 调整y的通道数使其与x匹配
#         if y.size(1) != x.size(1):
#             y = self.channel_adjust(y, x.size(1))
#
#         # 调整空间尺寸并相加
#         return F.interpolate(y, size=(H, W), mode='nearest') + x
# ####################todo gzl
# # 添加通道调整层
#     def channel_adjust(self, x, target_channels):
#      if x.size(1) != target_channels:
#         return nn.Conv2d(x.size(1), target_channels, kernel_size=1).to(x.device)(x)
#      return x
