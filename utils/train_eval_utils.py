import math
import sys
import time

import torch
import numpy as np
#import transforms_higherHRnet as transforms
import transforms  #Litepose和LAnet使用transforms
#import transforms_LightOpenPose as transforms  #LightOpenPose使用transforms_LightOpenPose
import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from .loss import KpLoss_for_val
from .loss import KpLoss_for_train
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_pro,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # lr_scheduler = None
    # if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
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

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images)

            losses = mse(results,targets)

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict({"losses": losses})
        print(losses)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
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

        # if lr_scheduler is not None:  # 第一轮使用warmup训练方式
        #     lr_scheduler.step()
        

        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        #breakpoint()
        # if i >= 2:
        # torch.save(model.state_dict(), '/home/ubuntu/文档/python/leaf_angle/HRNet/111/2.pth')
        # sys.exit()

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, loss_pro, flip=False, flip_pairs=None):
    # 读取真实值文件
    df = pd.read_excel('/home/ubuntu/桌面/qy/leafangle/train_data_5.8_5.9_23.12.11_12.15标注的文件_郑老师挑选_乔义修改.xlsx')
    Test = pd.read_excel('/home/ubuntu/桌面/标注文件/20240710ImagejLabel_true_leafangle.xlsx')
    df = df.sort_values(by='img_name')
    df = pd.concat([df, Test])
    if flip:
        assert flip_pairs is not None, "enable flip must provide flip_pairs."

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
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
        #计算叶夹角和R2
        batch_size, num_joints, h, w = outputs.shape
        heatmaps_reshaped = outputs.reshape(batch_size, num_joints, -1)
        
        maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
        maxvals = maxvals[0]
        maxvals = maxvals.cpu().detach().numpy()
        # maxvals = maxvals.unsqueeze(dim=-1)
        idx = idx.float()
        x_v = idx[0] % w  # column 对应最大值的x坐标
        y_v = torch.floor(idx[0] / w)  # row 对应最大值的y坐标
        #torch.cat将数据拼接，dim代表拼接的维度，0代表行，1代表列
        preds = torch.cat((x_v, y_v), dim=0)
        predict = preds.clone()
        predict = predict*torch.tensor(2)

        # 计算向量 A 和 B 的点乘
        if predict[5]-predict[4]<0:
            vector_A = [predict[2]-predict[1], predict[5]-predict[4]]
        else:
            vector_A = [predict[1]-predict[2], predict[4]-predict[5]]
        vector_B = [predict[0]-predict[1], predict[3]-predict[4]]
        dot_product = sum(a * b for a, b in zip(vector_A, vector_B))

        # 计算向量的长度
        magnitude_A = math.sqrt(sum(a**2 for a in vector_A))
        magnitude_B = math.sqrt(sum(b**2 for b in vector_B))

        # 计算夹角的弧度值
        cosine_similarity = dot_product / (magnitude_A * magnitude_B)
        angle_radians = math.acos(cosine_similarity) #返回x的反余弦弧度值

        # 将弧度转换为度数
        angle_degrees = math.degrees(angle_radians)
         # 将计算出的角度保存到predsall列表
        predsall.append(angle_degrees)
        imgPath= targets[0].get('img_path', None)  #读取文件名称
        imgPaths.append(imgPath)
        gts.extend(df.loc[df['img_path'] == imgPath, 't_angle'].values)

        if flip: #翻转图片再运行一次，从而综合两张图片的预测结果
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
        
        target_keypoint = np.concatenate([target['keypoints_new'] for target in targets], axis=0).reshape((len(targets), 3, 2))#这里将4改成了3，在计算3个点的时候
        # with open('/home/ubuntu/文档/python/leaf_angle/HRNet/save_weights/target_ouput.txt', "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     txt1 = np.array2string(target_keypoint, separator=', ')
        #     txt2 = np.array2string(outputs1[0], separator=', ')
        #     f.write(txt1 + "\n")
        #     f.write(txt2 + "\n")
        # print('target:', target_keypoint, 'outputs1[0]:', outputs1[0])
        e = np.sum((target_keypoint - outputs1[0])**2, axis=-1)/(2*images.shape[2]*images.shape[3]+np.spacing(1))#表示每个点的分数
        oks_all += np.sum(np.exp(-e))/(e.shape[-1])

        metric_logger.update(model_time=model_time)

    
        i+=1

    oks = oks_all/i
    mloss = loss_all/i
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    #计算叶夹角和R2
    # calc r2
    r2 = -1  # 设置默认值为 -1（表示无法计算的情况）
    rmse=-1
    if not np.isnan(predsall).any():
        r2 = r2_score(gts, predsall)  # 如果不存在 NaN，计算 r2
        mse = mean_squared_error(gts, predsall)
        rmse = np.sqrt(mse)
    # dic=pd.DataFrame({'img_path':imgPaths,'gt':gts,'pred':predsall})
    # dic.to_csv('/home/ubuntu/文档/python/leaf_angle/HRNet/result_5.9_1210.csv',index=False)
    # save_files = {
    #         'model': model.state_dict()
    # }
    # torch.save(save_files, "/home/ubuntu/文档/python/leaf_angle/HRNet/save_weights/test2.pth")
    return [oks, mloss],r2,rmse
