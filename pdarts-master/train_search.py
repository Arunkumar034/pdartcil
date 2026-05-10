import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype

# ---------- Custom IDX Dataset (MNIST-like) ----------
class IDXDataset(torch.utils.data.Dataset):
    """Reads IDX format files (e.g., MNIST). Assumes:
       - images file: 'train-images-idx3-ubyte' or given name
       - labels file: 'train-labels-idx1-ubyte' or given name
    """
    def __init__(self, root, images_file, labels_file, transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self._load_data(images_file, labels_file)

    def _load_data(self, images_file, labels_file):
        import struct
        # Load images
        img_path = os.path.join(self.root, images_file)
        with open(img_path, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2051:
                raise ValueError(f"Invalid image file magic number: {magic}")
            rows, cols = struct.unpack(">II", f.read(8))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
        # Load labels
        lbl_path = os.path.join(self.root, labels_file)
        with open(lbl_path, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid label file magic number: {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        from PIL import Image
        img = Image.fromarray(img, mode='L')  # 'L' = grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------------------------------

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
# Custom dataset arguments
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--in_channels', type=int, default=1, help='input image channels (1 for grayscale)')
parser.add_argument('--train_images_file', type=str, default='train-images-idx3-ubyte', help='train images file name')
parser.add_argument('--train_labels_file', type=str, default='train-labels-idx1-ubyte', help='train labels file name')
parser.add_argument('--test_images_file', type=str, default='t10k-images-idx3-ubyte', help='test images file name')
parser.add_argument('--test_labels_file', type=str, default='t10k-labels-idx1-ubyte', help='test labels file name')

args = parser.parse_args()

# ---------- Single fixed save directory (no timestamp) ----------
args.save = '{}search-{}'.format(args.save, args.note)
os.makedirs(args.save, exist_ok=True)

# ---------- Single log file setup (append mode) ----------
log_format = '%(asctime)s %(message)s'
log_file = os.path.join(args.save, 'log.txt')

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt='%m/%d %I:%M:%S %p',
    handlers=[
        logging.StreamHandler(sys.stdout),            # print to console
        logging.FileHandler(log_file, mode='a')       # append to single log file
    ]
)

# ----------------------------------------------------------------
# FIX: safe topk — never request k larger than number of classes
# ----------------------------------------------------------------
def get_topk(num_classes):
    """Returns (1, min(5, num_classes)) as the topk tuple.
    Prevents RuntimeError when num_classes < 5 (e.g. binary classification).
    """
    return (1, min(5, num_classes))


def main():
    # Log a separator so each run is easy to find in the single log file
    logging.info('=' * 60)
    logging.info('NEW RUN  %s', time.strftime("%Y%m%d-%H%M%S"))
    logging.info('=' * 60)

    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    # ---------- Prepare dataset (IDX format) ----------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = IDXDataset(
        root=args.tmp_data_dir,
        images_file=args.train_images_file,
        labels_file=args.train_labels_file,
        transform=train_transform
    )

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)

    # ---------- Build Network ----------
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)

    # Stage configuration
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]

    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]

    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]

    if len(args.dropout_rate) == 3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]

    eps_no_archs = [10, 10, 10]

    for sp in range(len(num_to_keep)):
        model = Network(
            C=args.init_channels + int(add_width[sp]),
            num_classes=args.num_classes,
            layers=args.layers + int(add_layers[sp]),
            criterion=criterion,
            switches_normal=switches_normal,
            switches_reduce=switches_reduce,
            p=float(drop_rate[sp]),
            in_channels=args.in_channels
        )
        model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)

        optimizer = torch.optim.SGD(
            network_params,
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        optimizer_a = torch.optim.Adam(
            model.module.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2

        for epoch in range(epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()

            if epoch < eps_no_arch:
                model.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.module.update_p()
                train_acc, train_obj = train(
                    train_queue, valid_queue, model, network_params,
                    criterion, optimizer, optimizer_a, lr, train_arch=False)
            else:
                model.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model.module.update_p()
                train_acc, train_obj = train(
                    train_queue, valid_queue, model, network_params,
                    criterion, optimizer, optimizer_a, lr, train_arch=True)

            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)

            if epochs - epoch < 5:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                logging.info('Valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])

        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)

        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()

        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False

        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()

        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False

        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce)

        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]

            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])

            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1

            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False

            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)

            logging.info('Restricting skipconnect...')
            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk = check_sk_number(switches_normal)
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)


def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    valid_queue_iter = iter(valid_queue)

    # FIX: safe topk — avoids RuntimeError when num_classes < 5
    topk = get_topk(args.num_classes)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        if train_arch:
            try:
                input_search, target_search = next(valid_queue_iter)
            except StopIteration:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        # FIX: use safe topk instead of hardcoded (1, 5)
        prec1, prec5 = utils.accuracy(logits, target, topk=topk)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    # FIX: safe topk — avoids RuntimeError when num_classes < 5
    topk = get_topk(args.num_classes)

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        # FIX: use safe topk instead of hardcoded (1, 5)
        prec1, prec5 = utils.accuracy(logits, target, topk=topk)
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def parse_network(switches_normal, switches_reduce):
    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)
    concat = range(2, 6)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    return index


def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    zf = (0 in idxs)
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index


def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:   # index 3 is 'skip_connect' in PRIMITIVES
            count = count + 1
    return count


def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            return -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
            return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if idx != -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0
    return probs_out


def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches


def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if i not in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)