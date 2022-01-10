from utils.layers import *
from utils.parse_config import *
from collections import OrderedDict

ONNX_EXPORT = False


def create_modules(module_defs, img_size, cfg):
    # Constructs module list of layer blocks from module configuration in module definitions

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            if any(x in cfg for x in ['panet', 'yolov4', 'cd53']):  # stride order reversed
                stride = list(reversed(stride))
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                # If previous layer is a dropout layer, get the one before
                if module_list[j].__class__.__name__ == 'Dropout':
                    j -= 1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        elif mdef['type'] == 'dropout':
            perc = float(mdef['probability'])
            modules = nn.Dropout(p=perc)
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (1)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    # YOLOv3 object detection model: https://github.com/AlexeyAB/darknet

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_once(x)
        else:  
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''
        # Augment images 
        if augment:  
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw

class Vgg16Base(nn.Module):

    def __init__(self, channels, classes, filters):
        super(Vgg16Base, self).__init__()

        self.conv_1_1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_1_2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_1 = nn.BatchNorm2d(filters)
        self.max_pool_1 = nn.MaxPool2d(2,2)

        self.conv_2_1 = nn.Conv2d(filters, filters*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_2_2 = nn.Conv2d(filters*2, filters*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_2 = nn.BatchNorm2d(filters*2)
        self.max_pool_2 = nn.MaxPool2d(2,2)

        self.conv_3_1 = nn.Conv2d(filters*2, filters*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3_2 = nn.Conv2d(filters*4, filters*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_3 = nn.BatchNorm2d(filters*4)
        self.max_pool_3 = nn.MaxPool2d(2,2)

        self.conv_4_1 = nn.Conv2d(filters*4, filters*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_4_2 = nn.Conv2d(filters*8, filters*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_4_3 = nn.Conv2d(filters*8, filters*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_4 = nn.BatchNorm2d(filters*8)
        self.max_pool_4 = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = F.elu(self.conv_1_1(x))
        x = self.conv_1_2(x)
        x = self.bn_1(x)
        x = F.elu(self.max_pool_1(x))

        x = F.elu(self.conv_2_1(x))
        x = self.conv_2_2(x)
        x = self.bn_2(x)
        x = F.elu(self.max_pool_2(x))

        x = F.elu(self.conv_3_1(x))
        x = self.conv_3_2(x)
        x = self.bn_3(x)
        x = F.elu(self.max_pool_3(x))

        x = F.elu(self.conv_4_1(x))
        x = F.elu(self.conv_4_2(x))
        x = self.conv_4_3(x)
        x = self.bn_4(x)
        x = F.elu(self.max_pool_4(x))

        return x


class SingleVgg16(nn.Module):

    def __init__(self, channels, classes, filters):
        super(SingleVgg16, self).__init__()
        self.vgg16Base = Vgg16Base(channels, classes, filters)

        self.conv_5_1 = nn.Conv2d(filters*8, filters*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_2 = nn.Conv2d(filters*8, filters*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_3 = nn.Conv2d(filters*8, filters*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_5 = nn.BatchNorm2d(filters*8)
        self.max_pool_5 = nn.MaxPool2d(2,2)

        self.fc_1 = nn.Linear(8 * filters * 8 * 8, 1000)
        self.bn_6 = nn.BatchNorm1d(1000)
        self.fc_2 = nn.Linear(1000, 500)
        self.bn_7 = nn.BatchNorm1d(500)
        self.fc_3 = nn.Linear(500, classes)


    def forward(self, x):
        x = self.vgg16Base(x)

        x = F.elu(self.conv_5_1(x))
        x = F.elu(self.conv_5_2(x))
        x = self.conv_5_3(x)
        x = self.bn_5(x)
        x = F.elu(self.max_pool_5(x))

        x = x.view(x.shape[0], -1)
        x = F.elu(self.fc_1(x))
        x = self.bn_6(x)
        x = F.elu(self.fc_2(x))
        x = self.bn_7(x)
        x = self.fc_3(x)
        return x

class UNet(nn.Module):

    def __init__(self, in_channels = 1, out_channels = 2, init_features = 16):
        super(UNet, self).__init__()

        channels = init_features

        self.encoder1 = UNet.conv_block(in_channels, channels, layer_name='encode1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet.conv_block(channels, channels * 2, layer_name="encode2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet.conv_block(channels * 2, channels * 4, layer_name="encode3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet.conv_block(channels * 4, channels * 8, layer_name="encode4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = UNet.conv_block(channels * 8, channels * 16, layer_name="encode5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet.conv_block(channels * 16, channels * 32, layer_name="bottleneck")

        self.upconv5 = nn.ConvTranspose2d(channels * 32, channels * 16, kernel_size=2, stride=2)
        self.decoder5 = UNet.conv_block(int((channels * 16) * 2.5), channels * 16, layer_name="decode5")
        self.upconv4 = nn.ConvTranspose2d(channels * 16, channels * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet.conv_block(int((channels * 8) * 2.5), channels * 8, layer_name="decode4")

        self.upconv3 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet.conv_block(int((channels * 4) * 2.5), channels * 4, layer_name="decode3")
        self.upconv2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet.conv_block(int((channels * 2) * 2.5), channels * 2, layer_name="decode2")
        self.upconv1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
        self.decoder1 = UNet.conv_block(channels * 2, channels, layer_name="decode1")

        self.conv = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=1)

    def forward(self, input):

        enc1_conv = self.encoder1(input)
        enc1_pool = self.pool1(enc1_conv)

        enc2_conv = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2_conv)

        enc3_conv = self.encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3_conv)

        enc4_conv = self.encoder4(enc3_pool)
        enc4_pool = self.pool4(enc4_conv)

        enc5_conv = self.encoder5(enc4_pool)
        enc5_pool = self.pool5(enc5_conv)

        bottleneck = self.bottleneck(enc5_pool)

        dec5_upcon = self.upconv5(bottleneck)
        dec5_cat = torch.cat((dec5_upcon, enc5_conv, enc4_pool), dim=1)
        dec5_conv = self.decoder5(dec5_cat)

        dec4_upcon = self.upconv4(dec5_conv)
        dec4_cat = torch.cat((dec4_upcon, enc4_conv, enc3_pool), dim=1)
        dec4_conv = self.decoder4(dec4_cat)

        dec3_upcon = self.upconv3(dec4_conv)
        dec3_cat = torch.cat((dec3_upcon, enc3_conv, enc2_pool), dim=1)
        dec3_conv = self.decoder3(dec3_cat)

        dec2_upcon = self.upconv2(dec3_conv)
        dec2_cat = torch.cat((dec2_upcon, enc2_conv, enc1_pool), dim=1)
        dec2_conv = self.decoder2(dec2_cat)

        dec1_upcon = self.upconv1(dec2_conv)
        dec1_cat = torch.cat((dec1_upcon, enc1_conv), dim=1)
        dec1_conv = self.decoder1(dec1_cat)

        ret = torch.sigmoid(self.conv(dec1_conv))

        return ret


    @staticmethod
    def conv_block(in_channels, features, layer_name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        layer_name + "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (layer_name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (layer_name + "relu1", nn.ReLU(inplace=True)),
                    (
                        layer_name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (layer_name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (layer_name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
