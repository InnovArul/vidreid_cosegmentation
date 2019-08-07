from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import more_itertools as mit

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms as T
import models
import utils
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint, get_features
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from project_utils import *

parser = argparse.ArgumentParser(
    description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--seq-len', type=int, default=4,
                    help="number of images to sample in a tracklet")
parser.add_argument('--test-num-tracks', type=int, default=16,
                    help="number of tracklets to pass to GPU during test (to avoid OOM error)")
# Optimization options
parser.add_argument('--max-epoch', default=800, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--data-selection', type=str,
                    default='random', help="random/evenly")
parser.add_argument('--train-batch', default=28, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='se_resnet50_cosam45_tp',
                    help="se_resnet50_tp, se_resnet50_ta")

# Miscs
parser.add_argument('--print-freq', type=int,
                    default=78, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str,
                    default=None, help='need to be set for loading pretrained models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=200,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-step', type=int, default=50)
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--save-prefix', type=str, default='se_resnet50_cosam45_tp')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def main():
    args.save_dir = args.arch + '_' + args.save_dir
    args.save_prefix = args.arch + '_' + args.save_dir

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    # append date with save_dir
    args.save_dir = '../scratch/' + utils.get_currenttime_prefix() + '_' + \
        args.dataset + '_' + args.save_dir
    if args.pretrained_model is not None:
        args.save_dir = os.path.dirname(args.pretrained_model)

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args.seq_len,
                     sample=args.data_selection, transform=transform_train),
        sampler=RandomIdentitySampler(
            dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len,
                     sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len,
                     sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, seq_len=args.seq_len)

    # pretrained model loading
    if args.pretrained_model is not None:
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(
                args.pretrained_model))
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        pretrained_state = torch.load(args.pretrained_model)['state_dict']
        print(len(pretrained_state), ' keys in pretrained model')

        current_model_state = model.state_dict()
        pretrained_state = {key: val
                            for key, val in pretrained_state.items()
                            if key in current_model_state and val.size() == current_model_state[key].size()}

        print(len(pretrained_state),
              ' keys in pretrained model are available in current model')
        current_model_state.update(pretrained_state)
        model.load_state_dict(current_model_state)

    print("Model size: {:.5f}M".format(sum(p.numel()
                                           for p in model.parameters())/1000000.0))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion_xent = CrossEntropyLabelSmooth(
        num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, args.pool, use_gpu)
        return

    start_time = time.time()
    best_rank1 = -np.inf

    is_first_time = True
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))

        train(model, criterion_xent, criterion_htri,
              optimizer, trainloader, use_gpu)

        if args.stepsize > 0:
            scheduler.step()

        rank1 = 'NA'
        is_best = False

        if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1

        # save the model as required
        if (epoch+1) % args.save_step == 0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, args.save_prefix + 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

        is_first_time = False
        if not is_first_time:
            utils.disable_all_print_once()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    import time
    torch.backends.cudnn.benchmark = True

    start = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(tqdm(trainloader)):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        imgs, pids = Variable(imgs), Variable(pids)

        outputs, features = model(imgs)

        # combine hard triplet loss with cross entropy loss
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(features, pids)
        loss = xent_loss + htri_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))
        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx +
                                                              1, len(trainloader), losses.val, losses.avg))


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    print('extracting query feats')
    for batch_idx, (imgs, pids, camids) in enumerate(tqdm(queryloader)):
        if use_gpu:
            imgs = imgs.cuda()

        with torch.no_grad():
            #imgs = Variable(imgs, volatile=True)
            # b=1, n=number of clips, s=16
            b, n, s, c, h, w = imgs.size()
            assert(b == 1)
            imgs = imgs.view(b*n, s, c, h, w)
            features = get_features(model, imgs, args.test_num_tracks)
            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            torch.cuda.empty_cache()

    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    print('extracting gallery feats')
    for batch_idx, (imgs, pids, camids) in enumerate(tqdm(galleryloader)):
        if use_gpu:
            imgs = imgs.cuda()

        with torch.no_grad():
            #imgs = Variable(imgs, volatile=True)
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b*n, s, c, h, w)
            assert(b == 1)
            # handle chunked data
            features = get_features(model, imgs, args.test_num_tracks)
            features = torch.mean(features, 0)
            torch.cuda.empty_cache()

        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    # re-ranking
    from person_re_ranking.python_version.re_ranking_feature import re_ranking
    rerank_distmat = re_ranking(
        qf.numpy(), gf.numpy(), k1=20, k2=6, lambda_value=0.3)
    print("Computing CMC and mAP for re-ranking")
    cmc, mAP = evaluate(rerank_distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()
