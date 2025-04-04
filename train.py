import os
import datetime
import random
import torch
from torch.utils import data
import numpy as np
import argparse
from nets.LANet import LANet
from utils.DatasetSeg import LeafKeypoint
import utils.transforms as transforms
from utils import utils


def get_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

#get_random_seed(66)
get_random_seed(66)
def create_model(load_pretrain_weights=False):
    model = LANet()
    # weights = torch.load(weights_path)
    # weights = weights if "model" not in weights else weights["model"]
    # model.load_state_dict(weights)
    if load_pretrain_weights: 
        #weights_path = './                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      weights/save_weights/every_epoch_model_20231226_144450.pth'
        #weights = torch.load(weights_path, map_location='cuda:0')
        weights = weights if "model" not in weights else weights["model"]
        model.load_state_dict(weights, strict=True)
    return model


def main(args):
    # print('args.loss_progress:', args.loss_progress)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # print("Using {} device training.".format(device.type)) 

    time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = "./weights/save_weights/results{}.txt".format(time_now)

    fixed_size = args.fixed_size

    heatmap_hw = (args.fixed_size[0] // args.heatmapscale, args.fixed_size[1] // args.heatmapscale)
    data_transform = {
        "train": transforms.Compose([
            # transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            transforms.AffineTransform(scale=(0.9, 1.3), rotation=(-15, 15), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5),#, person_kps_info["flip_pairs"]
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2),
            transforms.Blur(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5083, 0.5155, 0.5032], std=[0.1586, 0.1610, 0.1442])
            transforms.Normalize(mean=[0.501642, 0.517964,0.514330], std=[0.139957, 0.155858, 0.153716])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1, 1), fixed_size=fixed_size),# FIXME 
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5083, 0.5155, 0.5032], std=[0.1586, 0.1610, 0.1442])
            transforms.Normalize(mean=[0.501642	, 0.517964,0.514330], std=[0.139957, 0.155858, 0.153716])
        ])
    }

    train_dataset = LeafKeypoint(train= 'train', transforms=data_transform["train"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    val_dataset = LeafKeypoint(train= 'val', transforms=data_transform["val"])
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)
    


    # create model
    model = create_model()

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay
                                  #betas=(0.97, 0.999)
                                  )
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler  #xian zai shi CosineAnnealingLR
    if args.scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.scheduler == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)#0.0000001


    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    #best_oks =  0.0
    minloss = 1e5
    max_R_square=0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        loss_progress = False
        if args.loss_progress:
            loss_progress = True

        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=98, warmup=True,
                                              scaler=scaler, loss_pro = loss_progress)

        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
        # #ff0f0f
        # if epoch == 50:
        #     for name, param in model.named_parameters():
        #         if any(layer_name in name for layer_name in ['first', 'stage', 'attention', 'deconv_refined', 'deconv_raw']):
        #             param.requires_grad = True

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device,
                                   flip=False, flip_pairs=None, loss_pro = loss_progress)

        # write into txt
        with open(results_file, "a") as f:

            #result_info = [f"{i:.13f}" for i in coco_info[1] + [mean_loss.item()]] + [f"{lr:.6f}"]
            result_info = [f"{coco_info[1]:.13f}", f"{mean_loss.item():.13f}", f"{lr:.6f}"]

            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # val_map.append(coco_info[1])  # @0.5 mAP
        print('train_mloss:{:.10f}\nval_mloss:{:.10f}\n'.format(mean_loss.item(), coco_info[1]))
        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}

        torch.save(save_files, "./weights/save_weights/every_epoch_model_{}.pth".format(time_now))

        if minloss > coco_info[1]:
            minloss = coco_info[1]
            torch.save(save_files, "./weights/save_weights/min_loss_{}.pth".format(time_now))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='device')
    #NOTE Setting the path
    parser.add_argument('--data-path', default='./', help='dataset')

    #NOTE Setting key point information
    parser.add_argument('--person-det', type=str, default=None)
    # NOTE image size
    parser.add_argument('--fixed-size', default=[768, 576], nargs='+', type=int, help='input size')
    # If you need to continue the last training, specify the address of the last training weights file.
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # Designate the number of epochs to start training from.
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # NOTE epoch
    parser.add_argument('--epochs', default=500, type=int, metavar='N',# 170
                        help='number of total epochs to run')

    # NOTE learning rate
    #parser.add_argument('--lr-steps', default=[100, 200, 300, 400], nargs='+', type=int, help='decrease lr every step-size epochs')# 170-200
    #parser.add_argument('--lr-gamma', default=0.5, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr', default=0.002, type=float,
                        help='initial learning rate')
    
    parser.add_argument('--heatmapscale', default=2, type=float,
                        help='scale ratio of heatmap')
    
    # weight_decay
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # NOTE batch size
    parser.add_argument('--batch-size', default=4
                        , type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--loss-progress', default=False, type=bool, metavar='N',
                        help='loss progress when epoch equal to 60.')
    # training strategy
    parser.add_argument('--scheduler', default='CosineAnnealingLR', type=str, metavar='N',
                        help='training strategy.')
    # Whether to train with mixed precision (requires GPU support for mixed precision)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)


    main(args)
