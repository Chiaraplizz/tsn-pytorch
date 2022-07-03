import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser

best_prec1 = 0
np.random.seed(13696641)
torch.manual_seed(13696641)




def main():
    global args, best_prec1
    args = parser.parse_args()
    args.weight_i3d = '/home/chiarap/TSN/i3d/rgb_imagenet.pt'

    if args.dataset == 'small':
        num_class = 5
    elif args.dataset == 'full':
        num_class = 12
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    source = args.shift.split("-")[0]
    target = args.shift.split("-")[1]

    if source == 'U' and target == 'H':
        if args.dataset == 'small':
            args.train_list = "/home/chiarap/UCF_small/list_ucf101_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/HMDB_small/list_hmdb51_val_hmdb_ucf-feature.txt"
        else:
            args.train_list = "/home/chiarap/UCF_full/list_ucf101_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/HMDB_full/list_hmdb51_val_hmdb_ucf-feature.txt"
    elif source == 'H' and target == 'U':
        if args.dataset == 'small':
            args.train_list = "/home/chiarap/HMDB_small/list_hmdb51_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/UCF_small/list_ucf101_val_hmdb_ucf-feature.txt"
        else:
            args.train_list = "/home/chiarap/HMDB_full/list_hmdb51_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/UCF_full/list_ucf101_val_hmdb_ucf-feature.txt"

    if source == 'H' and target == 'H':
        if args.dataset == 'small':
            args.train_list = "/home/chiarap/HMDB_small/list_hmdb51_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/HMDB_small/list_hmdb51_val_hmdb_ucf-feature.txt"
        else:
            args.train_list = "/home/chiarap/HMDB_full/list_hmdb51_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/HMDB_full/list_hmdb51_val_hmdb_ucf-feature.txt"

    if source == 'U' and target == 'U':
        if args.dataset == 'small':
            args.train_list = "/home/chiarap/UCF_small/list_ucf101_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/UCF_small/list_ucf101_val_hmdb_ucf-feature.txt"
        else:
            args.train_list = "/home/chiarap/UCF_full/list_ucf101_train_hmdb_ucf-feature.txt"
            args.val_list = "/home/chiarap/UCF_full/list_ucf101_val_hmdb_ucf-feature.txt"

    model = TSN(args, num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.train_target:
        args.start_epoch = 0

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # todo: add train loader target
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB",
                                                                      "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB",
                                                                      "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size // 8, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1 or args.validate:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


# todo: add iter of training loader target

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()

    accum_iter = args.total_batch // args.batch_size
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        loss = 0
        # compute output
        if not args.train_target:
            loss = criterion(output, target_var)
            loss = loss / accum_iter
            # measure accuracy and record loss
            prec1, prec5, class_correct, class_total = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0), class_correct, class_total)
            top5.update(prec5.item(), input.size(0), class_correct, class_total)

        if args.entropy > 0:
            entropy_loss = entropy(output).mean()
            if args.g_ent:
                msoftmax = output.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gentropy_loss
            loss_e = entropy_loss * args.entropy
            loss += loss_e

        # compute gradient and do SGD step

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
        if (i + 1) % accum_iter == 0:
            if not args.train_target:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
            elif args.entropy:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Loss {loss:.4f} ({loss:.4f})\t'
                       'Loss_e {top1:.3f} ({top1:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=loss, top1=loss_e, lr=optimizer.param_groups[-1]['lr'])))

            elif args.entropy and args.g_ent:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Loss_g_ent {loss:.4f} ({loss:.4f})\t'
                       'Loss_e {top1:.3f} ({top1:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=gentropy_loss, top1=loss_e, lr=optimizer.param_groups[-1]['lr'])))


def entropy(x):
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    entropy_ = torch.sum(-softmax(x) * logsoftmax(x), 1)
    return entropy_


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    class_accuracies = list()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        input_var = input_var.cuda()
        B, C, H, W = input_var.shape
        input_var = input_var.reshape(B, 4, -1, H, W)
        input_var = input_var.permute(1, 0, 2, 3, 4)

        outputs = torch.zeros(4, B, 12).cuda()
        for i_s, seg in enumerate(input_var):
            out = model(input=seg)
            outputs[i_s, :, :] = out

        output = torch.mean(outputs, dim=0)  # fai la media dei logits dei 5 segmenti
        # compute output
        # output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5, c_c, c_t = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0), c_c, c_t)
        top5.update(prec5.item(), input.size(0), c_c, c_t)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5)))
    class_accuracies.append([(x / y) * 100 for x, y in zip(top1.correct, top1.total)])
    print('----- END model!\ttop1 = %.2f%%\ttop5 = %.2f%%\nCLASS accuracy:' % (top1.avg, top5.avg))
    for i_class, class_acc in enumerate(class_accuracies[0]):
        print('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                               int(top1.correct[i_class]),
                                               int(top1.total[i_class]),
                                               class_acc))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.correct = list(0. for _ in range(12))
        self.total = list(0. for _ in range(12))


    def update(self, val, n=1, class_correct = None, class_total = None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if class_correct is not None and class_total is not None:
            for i in range(0, 12):
                self.correct[i] += class_correct[i]
                self.total[i] += class_total[i]

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    class_correct, class_total = accuracy_per_class(correct[:1].view(-1), target, 12)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], res[1], class_correct, class_total

def accuracy_per_class(correct, target, num_label):
    class_correct = list(0. for _ in range(0, num_label))
    class_total = list(0. for _ in range(0, num_label))
    for i in range(0, target.size(0)):
        class_label = target[i].item()
        class_correct[class_label] += correct[i].item()
        class_total[class_label] += 1
    return class_correct, class_total

if __name__ == '__main__':
    main()
