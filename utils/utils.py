import math
import sys
import time

import torch
import numpy as np
import utils.transforms as transforms 
import pandas as pd
from collections import defaultdict, deque
import datetime
import time
import errno
import os

import torch
import torch.distributed as dist

class KpLoss_for_val(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        loss1 = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss1 = torch.sum(loss1) / bs

        return loss1
class KpLoss_for_train(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]

        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])


        loss1 = self.criterion(logits, heatmaps).mean(dim=[2, 3])


        loss1 = torch.sum(loss1) / bs

        return loss1

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # deque简单理解成加强版list
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):  # @property 是装饰器，这里可简单理解为增加median属性(只读)
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    收集各个进程中的数据
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  
    if world_size == 1:
        return [data]

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2: 
        return input_dict
    with torch.no_grad(): 
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}'])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_second = int(iter_time.global_avg * (len(iterable) - i))
                eta_string = str(datetime.timedelta(seconds=eta_second))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,
                                                         total_time / len(iterable)))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
       
        if x >= warmup_iters:  
            return 1
        alpha = float(x) / warmup_iters
      
        # print(x, warmup_factor * (1 - alpha) + alpha)
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
   
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_pro,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # lr_scheduler = None
    # if epoch == 0 and warmup is True: 
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)

    #     lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mse = KpLoss_for_train()
    mloss = torch.zeros(1).to(device)  # mean losses

    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.stack([image.to(device) for image in images])
        # print(images.dtype)
        # for ii in range(len(targets)):

        #     print(targets['img_path'])

       
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images)

            losses = mse(results,targets)

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = reduce_dict({"losses": losses})
        #print(losses)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
       
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        # current_lr = optimizer.param_groups[0]['lr']
        # print('current_lr:', current_lr)

        # if lr_scheduler is not None:
        #     lr_scheduler.step()
        

        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)


    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, loss_pro, flip=False, flip_pairs=None):
    # 读取真实值文件
    df = pd.read_excel('datasets/labels/labels.xlsx')
    df = df.sort_values(by='img_name')
    if flip:
        assert flip_pairs is not None, "enable flip must provide flip_pairs."

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    i = 0
    oks_all = 0
    loss_all = 0
    gts=[] #angle 
    predsall=[]
    imgPaths=[]
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack([img.to(device) for img in image])


        imgPath= targets[0].get('img_path', None)  #读取文件名称
        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        
        outputs = model(images)
        mse = KpLoss_for_val()
        losses = mse(outputs, targets)
        loss_all +=losses
        
        if flip:
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

        model_time = time.time() - model_time

        # decode keypoint
        reverse_trans = [t["reverse_trans"] for t in targets]
        outputs1 = transforms.get_final_preds(outputs, reverse_trans, post_processing=False)
        
        target_keypoint = np.concatenate([target['keypoints_new'] for target in targets], axis=0).reshape((len(targets), 3, 2))

        e = np.sum((target_keypoint - outputs1[0])**2, axis=-1)/(2*images.shape[2]*images.shape[3]+np.spacing(1))
        oks_all += np.sum(np.exp(-e))/(e.shape[-1])

        metric_logger.update(model_time=model_time)

    
        i+=1

    oks = oks_all/i
    mloss = loss_all/i
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return [oks, mloss]
