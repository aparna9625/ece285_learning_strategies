""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
#from visualize import plot

config = SearchConfig()

device = torch.device("cuda")

# tensorboard，支持简单的markdown文本格式记录
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)  # 这里记录的是 config 信息

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    print("gpus",config.gpus)
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get cifar10 and cifar100 data with meta info
    _, input1_channels, n1_classes, train1_data = utils.get_data(
        config.dataset1, config.data_path, cutout_length=0, validation=False)
    
    #first 10k datapoints
    train1_data.data = train1_data.data[:10000]
    print(train1_data)

#     _, input2_channels, n2_classes, train2_data = utils.get_data(
#         config.dataset2, config.data_path, cutout_length=0, validation=False)

#     assert len(train1_data) == len(train2_data)

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input1_channels, config.init_channels, n1_classes,
                                config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer for net1 and net2 (cifar10 and cifar100)
    w1_optim = torch.optim.SGD(model.weights(flag=config.dataset1), config.w_lr, momentum=config.w_momentum,
                               weight_decay=0.)
#     w2_optim = torch.optim.SGD(model.weights(flag=config.dataset2), config.w_lr, momentum=config.w_momentum,
#                                weight_decay=0.)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train1_data)
    split = n_train // 2
    indices = list(range(n_train))

    train1_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid1_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train1_loader = torch.utils.data.DataLoader(train1_data,
                                                batch_size=config.batch_size,
                                                sampler=train1_sampler,
                                                num_workers=config.workers,
                                                pin_memory=True)
    valid1_loader = torch.utils.data.DataLoader(train1_data,
                                                batch_size=config.batch_size,
                                                sampler=valid1_sampler,
                                                num_workers=config.workers,
                                                pin_memory=True)
    lr1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w1_optim, config.epochs, eta_min=config.w_lr_min)

#     train2_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
#     valid2_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
#     train2_loader = torch.utils.data.DataLoader(train2_data,
#                                                 batch_size=config.batch_size,
#                                                 sampler=train2_sampler,
#                                                 num_workers=config.workers,
#                                                 pin_memory=True)
#     valid2_loader = torch.utils.data.DataLoader(train2_data,
#                                                 batch_size=config.batch_size,
#                                                 sampler=valid2_sampler,
#                                                 num_workers=config.workers,
#                                                 pin_memory=True)
#     lr2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         w2_optim, config.epochs, eta_min=config.w_lr_min)

    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_loss = 9999999.
    best_sub1top1 = 0.
    best_sub2top1 = 0.
    for epoch in range(config.epochs):
        lr1_scheduler.step()
        lr1 = lr1_scheduler.get_lr()[0]
#         lr2_scheduler.step()
#         lr2 = lr2_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train_step(
            train1_loader, valid1_loader,
            model, architect, w1_optim, alpha_optim, lr1, epoch)

        # validation
        cur_step = (epoch + 1) * len(train1_loader)
        sub1top1, loss = validate(valid1_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        #plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        #caption = "Epoch {}".format(epoch + 1)
        #plot(genotype.normal, plot_path + "-normal", caption)
        #plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_loss > loss:
            best_loss = loss
            best_sub1top1 = sub1top1
#             best_sub2top1 = sub2top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec1@1 = {:.4%}".format(best_sub1top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train_step(train1_loader, valid1_loader, model, architect, w1_optim,
               alpha_optim, lr1, epoch):
    # top1 = utils.AverageMeter()
    # top5 = utils.AverageMeter()
    # losses = utils.AverageMeter()

    sub1top1 = utils.AverageMeter()
    sub1top5 = utils.AverageMeter()
    sub1losses = utils.AverageMeter()

#     sub2top1 = utils.AverageMeter()
#     sub2top5 = utils.AverageMeter()
#     sub2losses = utils.AverageMeter()

#     assert len(train1_loader) == len(train2_loader)
#     assert len(valid1_loader) == len(valid2_loader)
    cur_step = epoch * len(train1_loader)
    writer.add_scalar('train/lr1', lr1, cur_step)
#     writer.add_scalar('train/lr2', lr2, cur_step)

    weights1_path = os.path.join(config.path, 'net1_arch_weights.pth')
#     weights2_path = os.path.join(config.path, 'net2_arch_weights.pth')

    net1_weights, net2_weights = None, None
    # interleaving_weight_decay = 0.1  # [1, 0.1, 0.01, 0.001, 0.0001]

    model.train()

    for step, ((trn1_X, trn1_y), (val1_X, val1_y)) in enumerate(
            zip(train1_loader, valid1_loader)):
        trn1_X, trn1_y = trn1_X.to(device, non_blocking=True), trn1_y.to(device, non_blocking=True)
        val1_X, val1_y = val1_X.to(device, non_blocking=True), val1_y.to(device, non_blocking=True)
#         trn2_X, trn2_y = trn2_X.to(device, non_blocking=True), trn2_y.to(device, non_blocking=True)
#         val2_X, val2_y = val2_X.to(device, non_blocking=True), val2_y.to(device, non_blocking=True)

#         assert trn1_X.size(0) == trn2_X.size(0)
        N = trn1_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn1_X, trn1_y, val1_X, val1_y, lr1, w1_optim, net1_weights, flag=config.dataset1)
#         architect.unrolled_backward(trn2_X, trn2_y, val2_X, val2_y, lr2, w2_optim, net2_weights, flag=config.dataset2)
        alpha_optim.step()
        
#         if os.path.exists(weights2_path):
#             updated_state_dict, net1_weights = model.load_arch_weights(model.net1.state_dict(), weights2_path)
#             model.net1.load_state_dict(updated_state_dict)

        # phase 1. child network step (w)
        w1_optim.zero_grad()
        logits1 = model(trn1_X, flag=config.dataset1)
        loss1 = model.criterion(logits1, trn1_y)
        # l2 regularization on interleaving architecture weights
        for name, param in model.named_weights(flag=config.dataset1):
            if net1_weights is not None and name in net1_weights:
                loss1 += 0.5 * config.w_weight_decay * torch.pow((param - net1_weights[name]).norm(2), 2)
                # loss1 += 0.5 * interleaving_weight_decay * torch.pow((param - net1_weights[name]).norm(2), 2)
            else:
                loss1 += 0.5 * config.w_weight_decay * torch.pow(param.norm(2), 2)
        loss1.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(flag=config.dataset1), config.w_grad_clip)
        w1_optim.step()

        model.save_arch_weights(weights1_path, flag=config.dataset1)
#         updated_state_dict, net2_weights = model.load_arch_weights(model.net2.state_dict(), weights1_path)
#         model.net2.load_state_dict(updated_state_dict)

#         w2_optim.zero_grad()
#         logits2 = model(trn2_X, flag=config.dataset2)
#         loss2 = model.criterion(logits2, trn2_y)
#         # l2 regularization on interleaving architecture weights
#         for name, param in model.named_weights(flag=config.dataset2):
#             if net2_weights is not None and name in net2_weights:
#                 loss2 += 0.5 * config.w_weight_decay * torch.pow((param - net2_weights[name]).norm(2), 2)
#                 # loss2 += 0.5 * interleaving_weight_decay * torch.pow((param - net2_weights[name]).norm(2), 2)
#             else:
#                 loss2 += 0.5 * config.w_weight_decay * torch.pow(param.norm(2), 2)
#         loss2.backward()
#         nn.utils.clip_grad_norm_(model.weights(flag=config.dataset2), config.w_grad_clip)
#         w2_optim.step()

#         model.save_arch_weights(weights2_path, flag=config.dataset2)

        sub1prec1, sub1prec5 = utils.accuracy(logits1, trn1_y, topk=(1, 5))
        sub1losses.update(loss1.item(), N)
        sub1top1.update(sub1prec1.item(), N)
        sub1top5.update(sub1prec5.item(), N)

#         sub2prec1, sub2prec5 = utils.accuracy(logits2, trn2_y, topk=(1, 5))
#         sub2losses.update(loss2.item(), N)
#         sub2top1.update(sub2prec1.item(), N)
#         sub2top5.update(sub2prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train1_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss1 {sub1losses.avg:.3f} "
                "Prec1@(1,5) ({sub1top1.avg:.1%}, {sub1top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train1_loader) - 1,
                    sub1losses=sub1losses, sub1top1=sub1top1, sub1top5=sub1top5))

#         if step % config.print_freq == 0 or step == len(train2_loader) - 1:
#             logger.info(
#                 "Train: [{:2d}/{}] Step {:03d}/{:03d} "
#                 "Loss2 {sub2losses.avg:.3f} Prec2@(1,5) ({sub2top1.avg:.1%}, {sub2top5.avg:.1%})".format(
#                     epoch + 1, config.epochs, step, len(train2_loader) - 1,
#                     sub2losses=sub2losses, sub2top1=sub2top1, sub2top5=sub2top5))

        writer.add_scalar('train/sub1loss', loss1.item(), cur_step)
        writer.add_scalar('train/sub1top1', sub1prec1.item(), cur_step)
        writer.add_scalar('train/sub1top5', sub1prec5.item(), cur_step)
#         writer.add_scalar('train/sub2loss', loss2.item(), cur_step)
#         writer.add_scalar('train/sub2top1', sub2prec1.item(), cur_step)
#         writer.add_scalar('train/sub2top5', sub2prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec1@1 {:.4%}".format(epoch + 1, config.epochs, sub1top1.avg))

def validate(valid1_loader, model, epoch, cur_step):
    sub1top1 = utils.AverageMeter()
    sub1top5 = utils.AverageMeter()
    sub1losses = utils.AverageMeter()

#     sub2top1 = utils.AverageMeter()
#     sub2top5 = utils.AverageMeter()
#     sub2losses = utils.AverageMeter()

#     assert len(valid1_loader) == len(valid2_loader)

    model.eval()

    with torch.no_grad():
        for step, (X1, y1) in enumerate(valid1_loader):
            X1, y1 = X1.to(device, non_blocking=True), y1.to(device, non_blocking=True)
#             X2, y2 = X2.to(device, non_blocking=True), y2.to(device, non_blocking=True)

#             assert X1.size(0) == X2.size(0)
            N = X1.size(0)

            logits1 = model(X1, flag=config.dataset1)
            loss1 = model.criterion(logits1, y1)

#             logits2 = model(X2, flag=config.dataset2)
#             loss2 = model.criterion(logits2, y2)

            sub1prec1, sub1prec5 = utils.accuracy(logits1, y1, topk=(1, 5))
            sub1losses.update(loss1.item(), N)
            sub1top1.update(sub1prec1.item(), N)
            sub1top5.update(sub1prec5.item(), N)

#             sub2prec1, sub2prec5 = utils.accuracy(logits2, y2, topk=(1, 5))
#             sub2losses.update(loss2.item(), N)
#             sub2top1.update(sub2prec1.item(), N)
#             sub2top5.update(sub2prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid1_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss1 {sub1losses.avg:.3f} "
                    "Prec1@(1,5) ({sub1top1.avg:.1%}, {sub1top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid1_loader) - 1,
                        sub1losses=sub1losses, sub1top1=sub1top1, sub1top5=sub1top5))

#             if step % config.print_freq == 0 or step == len(valid2_loader) - 1:
#                 logger.info(
#                     "Valid: [{:2d}/{}] Step {:03d}/{:03d} "
#                     "Loss2 {sub2losses.avg:.3f} Prec2@(1,5) ({sub2top1.avg:.1%}, {sub2top5.avg:.1%})".format(
#                         epoch + 1, config.epochs, step, len(valid2_loader) - 1,
#                         sub2losses=sub2losses, sub2top1=sub2top1, sub2top5=sub2top5))

    writer.add_scalar('val/sub1loss', sub1losses.avg, cur_step)
    writer.add_scalar('val/sub1top1', sub1top1.avg, cur_step)
    writer.add_scalar('val/sub1top5', sub1top5.avg, cur_step)
#     writer.add_scalar('val/sub2loss', sub2losses.avg, cur_step)
#     writer.add_scalar('val/sub2top1', sub2top1.avg, cur_step)
#     writer.add_scalar('val/sub2top5', sub2top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec1@1 {:.4%}".format(epoch + 1, config.epochs, sub1top1.avg))

    return sub1top1.avg, sub1losses.avg

if __name__ == "__main__":
    main()
