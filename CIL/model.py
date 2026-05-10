import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

# ================= PDARTS DEFINITIONS =================

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# YOUR SEARCHED ARCHITECTURE
genotype = Genotype(
    normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('sep_conv_5x5', 4)],
    reduce_concat=range(2, 6)
)

# ================= OPERATIONS =================

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated Convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """Separable Convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class Zero(nn.Module):
    """Zero operation - returns zeros with same spatial size"""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    """Used for skip_connect when stride=2"""
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

OPS = {
    'none': lambda C, s, affine: Zero(s),
    'avg_pool_3x3': lambda C, s, affine: nn.AvgPool2d(3, stride=s, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, s, affine: nn.MaxPool2d(3, stride=s, padding=1),
    'skip_connect': lambda C, s, affine: nn.Identity() if s == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, s, affine: SepConv(C, C, 3, s, 1, affine=affine),
    'sep_conv_5x5': lambda C, s, affine: SepConv(C, C, 5, s, 2, affine=affine),
    'sep_conv_7x7': lambda C, s, affine: SepConv(C, C, 7, s, 3, affine=affine),
    'dil_conv_3x3': lambda C, s, affine: DilConv(C, C, 3, s, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, s, affine: DilConv(C, C, 5, s, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, s, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, s), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(s, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}

# ================= CELL =================

class Cell(nn.Module):
    def __init__(self, genotype, C_pp, C_p, C, reduction):
        super(Cell, self).__init__()
        
        # Preprocessing of inputs
        self.pre0 = ReLUConvBN(C_pp, C, 1, 1, 0)
        self.pre1 = ReLUConvBN(C_p, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        
        self._compile(C, op_names, indices, reduction, concat)

    def _compile(self, C, op_names, indices, reduction, concat):
        self._ops = nn.ModuleList()
        self._indices = indices
        self._concat = concat
        
        for name, idx in zip(op_names, indices):
            stride = 2 if reduction and idx < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)

    def forward(self, s0, s1):
        s0 = self.pre0(s0)
        s1 = self.pre1(s1)

        # Handle size mismatch
        if s0.size(2) != s1.size(2):
            s0 = F.interpolate(s0, size=s1.shape[2:], mode="nearest")

        states = [s0, s1]
        for i in range(0, len(self._ops), 2):
            h1 = self._ops[i](states[self._indices[i]])
            h2 = self._ops[i+1](states[self._indices[i+1]])
            # Ensure spatial dimensions match
            if h1.size(2) != h2.size(2):
                h1 = F.interpolate(h1, size=h2.shape[2:], mode="nearest")
            states.append(h1 + h2)

        return torch.cat([states[i] for i in self._concat], dim=1)

# ================= PDARTS BACKBONE =================

class PDARTSBackbone(nn.Module):
    def __init__(self, C=32, layers=6):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        C_pp, C_p, C_cur = C, C, C
        self.cells = nn.ModuleList()
        
        reductions = [layers // 3, 2 * layers // 3]

        for i in range(layers):
            reduction = i in reductions
            if reduction:
                C_cur *= 2

            cell = Cell(genotype, C_pp, C_p, C_cur, reduction)
            self.cells.append(cell)

            C_pp, C_p = C_p, C_cur * 4

        self.feature_dim = C_p
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)

        feat = self.gap(s1).view(s1.size(0), -1)
        return feat

# ================= EXPANDABLE FEATURE EXTRACTOR =================

class ExpandableFeatureExtractor(nn.Module):
    def __init__(self, C=32, layers=6):
        super().__init__()
        
        self.extractors = nn.ModuleList([PDARTSBackbone(C=C, layers=layers)])
        self.single_branch_dim = self.extractors[0].feature_dim
        self.out_dim = self.single_branch_dim

    def add_new_task_backbone(self):
        for ext in self.extractors:
            for p in ext.parameters():
                p.requires_grad = False

        new_ext = PDARTSBackbone(
            C=self.extractors[0].stem[0].out_channels,
            layers=len(self.extractors[0].cells)
        )
        self.extractors.append(new_ext)

        self.out_dim = len(self.extractors) * self.single_branch_dim

    def forward(self, x):
        feats = [ext(x) for ext in self.extractors]
        feats = torch.cat(feats, dim=1)
        return F.normalize(feats, dim=1)

# ================= CLASSIFIERS =================

class ExpandableClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def expand(self, new_in_dim, n_new_classes):
        old_w = self.fc.weight.data.clone()
        old_b = self.fc.bias.data.clone()

        old_out, old_in = old_w.shape
        
        new_fc = nn.Linear(new_in_dim, old_out + n_new_classes)

        nn.init.zeros_(new_fc.weight)
        nn.init.zeros_(new_fc.bias)

        with torch.no_grad():
            new_fc.weight[:old_out, :old_in] = old_w
            new_fc.bias[:old_out] = old_b

        self.fc = new_fc

    def forward(self, x):
        return self.fc(x)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_dim, n_new_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_new_classes + 1)

    def forward(self, x):
        return self.fc(x)