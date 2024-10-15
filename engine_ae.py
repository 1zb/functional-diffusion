# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (query_points, query_values, context_points, context_values, surface, categories) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        query_points = query_points.to(device, non_blocking=True)
        query_values = query_values.to(device, non_blocking=True)
        context_points = context_points.to(device, non_blocking=True)
        context_values = context_values.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)


        with torch.cuda.amp.autocast(enabled=False):

            loss_diff = model(
                context_points, 
                context_values,
                query_points, 
                query_values, 
                surface,
                categories,
            )
            
            loss = loss_diff#

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        # metric_logger.update(loss_recon=loss_recon.item())
        # metric_logger.update(weight=weight.item())
        # metric_logger.update(loss_bce_vol=loss_bce_vol.item())
        # metric_logger.update(loss_bce_near=loss_bce_near.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for query_points, query_values, context_points, context_values, surface, categories in metric_logger.log_every(data_loader, 50, header):

        query_points = query_points.to(device, non_blocking=True)
        query_values = query_values.to(device, non_blocking=True)
        context_points = context_points.to(device, non_blocking=True)
        context_values = context_values.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):

            loss_diff, loss_recon, weight, loss_bce_vol, loss_bce_near = model(
                context_points, 
                context_values,
                query_points, 
                query_values, 
                surface,
                categories,
            )
            
            # if loss_kl is not None:
            #     loss = loss_vol + 0.1 * loss_near + kl_weight * loss_kl
            # else:
            #     loss = loss_vol + 0.1 * loss_near
            loss = loss_diff#


        batch_size = surface.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_recon=loss_recon.item())
        metric_logger.update(weight=weight.item())
        metric_logger.update(loss_bce_vol=loss_bce_vol.item())
        metric_logger.update(loss_bce_near=loss_bce_near.item())

        # metric_logger.meters['iou'].update(iou.item(), n=batch_size)

        # if loss_kl is not None:
        #     metric_logger.update(loss_kl=loss_kl.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}