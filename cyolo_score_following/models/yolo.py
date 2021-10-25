"""
adapted from https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
"""
import json
import os

from copy import deepcopy
from cyolo_score_following.models.custom_modules import *
from cyolo_score_following.models.conditioning_networks import *
from cyolo_score_following.utils.general import make_divisible
from cyolo_score_following.utils.data_utils import FPS, SAMPLE_RATE, FRAME_SIZE


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                nn.init.constant_(param[m.hidden_size:m.hidden_size*2], 1)  # fg bias
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    if isinstance(m, nn.ELU):
        m.inplace = True


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=1, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc
        self.no = 5  # number of outputs per anchor (bbox and objectness)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):

        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.stride is not None:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # z.append(y.view(bs, -1, self.no))

                # add class idx for evaluation
                y = torch.cat((y, torch.ones_like(y[..., 0:1])*i), axis=-1)
                z.append(y.view(bs, -1, self.no + 1))

        if len(z) > 0:
            z = torch.cat(z, 1)
        return z, x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg, ch=1):  # model, input channels
        super(Model, self).__init__()

        self.yaml = cfg  # model dict

        self.zdim = self.yaml['encoder']['params']['zdim']
        self.nc = self.yaml['nc']
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist

        print(f"Conditioning: {self.yaml['encoder']['type']} | Parameters: {self.yaml['encoder']['params']}")
        self.conditioning_network = eval(self.yaml['encoder']['type'])(groupnorm=self.yaml['groupnorm'],
                                                                       **self.yaml['encoder']['params'])

        self.spec_module = LogSpectrogram(sr=SAMPLE_RATE, fps=FPS, frame_size=FRAME_SIZE)

        self.loss_type = self.yaml.get('loss_type', "mse")
        # Init weights, biases
        self.apply(initialize_weights)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.predict(torch.zeros(1, ch, s, s),
                                                                           torch.zeros(1, self.zdim))[1]])
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

    def forward(self, score, perf, tempo_aug=False):

        perf = self.compute_spec(perf, tempo_aug)
        z = self.encode_sequence(perf)[0]

        return self.predict(score, z)

    def encode_sequence(self, perf, hidden=None):
        return self.conditioning_network.encode_sequence(perf, hidden)

    def compute_spec(self, x, tempo_aug=False):
        return self.spec_module(x, tempo_aug)

    def predict(self, x, z):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if isinstance(m, FiLMConv):
                x = m(x, z)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output

        return x


def parse_model(d, ch):  # model_dict, input_channels(3)
    anchors, nc = d['anchors'], d['nc']
    activation = eval(d.get('activation', 'nn.ELU'))
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * 5  # number of outputs = anchors * 5 (5 bbox + objectness)
    groupnorm = d.get('groupnorm', False)
    zdim = d['encoder']['params']['zdim']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, m, args) in enumerate(d['backbone'] + d['head']):  # from,  module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        if m in [Conv, Focus, FiLMConv, Bottleneck]:
            c1, c2 = ch[f], args[0]

            c2 = make_divisible(c2 , 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]

        elif m in [nn.BatchNorm2d, nn.GroupNorm]:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        if m in [Conv,  Focus, Bottleneck]:
            m_ = m(*args, groupnorm=groupnorm, activation=activation)  # module
        elif m == FiLMConv:
            m_ = m(*args, zdim=zdim, groupnorm=groupnorm, activation=activation)  # module
        else:
            m_ = m(*args)  # module

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)

    model = nn.Sequential()

    for i, layer in enumerate(layers):
        model.add_module(f'{layer._get_name()}_{i}', layer)

    return model, sorted(save)


def build_model_and_criterion(config):
    from cyolo_score_following.utils.loss import compute_loss
    model = Model(config)
    criterion = compute_loss

    return model, criterion


def load_pretrained_model(param_path):
    param_dir = os.path.dirname(param_path)

    with open(os.path.join(param_dir, 'net_config.json'), 'r') as f:
        config = json.load(f)

    network, criterion = build_model_and_criterion(config)

    try:
        network.load_state_dict(torch.load(param_path, map_location=lambda storage, location: storage))
    except:
        network = nn.parallel.DataParallel(network)
        network.load_state_dict(torch.load(param_path, map_location=lambda storage, location: storage))
        network = network.module

    return network, criterion


if __name__ == '__main__':
    import argparse
    from cyolo_score_following.utils.general import load_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  help='model configuration file', default="configs/cyolo.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    # Create model
    model = Model(config)
    model.train()
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    inference, train = model(torch.zeros(4, 1, 416, 416), [torch.zeros(100000) for _ in range(4)])
    print(inference.shape, train[0].shape)
