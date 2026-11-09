import json
import os.path as osp
import time
import torch
import numpy as np
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import os
from pathlib import Path
# private package
from conf import *
from lib.dataset import AlignmentDataset, ImageDataset
from lib.backbone import StackedHGNetV1
from lib.loss import *
from lib.metric import NME, FR_AUC
from lib.utils import convert_secs2time
from lib.utils import AverageMeter


def get_config(args):
    config = None
    config_name = args.config_name
    if config_name == "alignment":
        config = Alignment(args)
    else:
        assert NotImplementedError

    return config


def get_dataset(config, tsv_file, image_dir, loader_type, is_train):
    dataset = None
    if loader_type == "alignment":
        dataset = AlignmentDataset(
            tsv_file,
            image_dir,
            transforms.Compose([transforms.ToTensor()]),
            config.width,
            config.height,
            config.channels,
            config.means,
            config.scale,
            config.classes_num,
            config.crop_op,
            config.aug_prob,
            config.edge_info,
            config.flip_mapping,
            is_train,
            encoder_type=config.encoder_type
        )
    elif loader_type =='test_image':
        dataset = ImageDataset(
            tsv_file,
            image_dir,
            transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256))]),
            config.width,
            config.height,
            config.channels,
            config.means,
            config.scale,
            config.classes_num,
            config.crop_op,
            config.aug_prob,
            config.edge_info,
            config.flip_mapping,
            is_train,
            encoder_type=config.encoder_type
        )
    else:
        assert False
    return dataset


def get_dataloader(config, data_type, world_rank=0, world_size=1):
    loader = None
    if data_type == "train":
        dataset = get_dataset(
            config,
            config.train_tsv_file,
            config.train_pic_dir,
            config.loader_type,
            is_train=True)
        if world_size > 1:
            sampler = DistributedSampler(dataset, rank=world_rank, num_replicas=world_size, shuffle=True)
            loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size // world_size,
                                num_workers=config.train_num_workers, pin_memory=True, drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=config.train_num_workers)
    elif data_type == "val":
        dataset = get_dataset(
            config,
            config.val_tsv_file,
            config.val_pic_dir,
            config.loader_type,
            is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.val_batch_size,
                            num_workers=config.val_num_workers)
    elif data_type == "test":
        dataset = get_dataset(
            config,
            config.test_tsv_file,
            config.test_pic_dir,
            config.loader_type,
            is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size,
                            num_workers=config.test_num_workers)
    elif data_type == 'test_image':
        dataset = get_dataset(
            config,
            config.test_tsv_file,
            config.test_pic_dir,
            'test_image',
            is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size,
                            num_workers=config.test_num_workers)
    else:
        assert False
    return loader


def get_optimizer(config, net):
    params = net.parameters()

    optimizer = None
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.learn_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.learn_rate)
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            params,
            lr=config.learn_rate,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
    else:
        assert False
    return optimizer


def get_scheduler(config, optimizer):
    # if config.scheduler == "MultiStepLR":
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    # else:
    #     assert False
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    return scheduler


def get_net(config):
    net = None
    if config.net == "stackedHGnet_v1":
        net = StackedHGNetV1(config=config,
                             classes_num=config.classes_num,
                             edge_info=config.edge_info,
                             nstack=config.nstack,
                             add_coord=config.add_coord,
                             decoder_type=config.decoder_type)
    else:
        assert False
    return net


def get_criterions(config):
    criterions = list()
    for k in range(config.label_num):
        if config.criterions[k] == "AWingLoss":
            criterion = AWingLoss()
        elif config.criterions[k] == "smoothl1":
            criterion = SmoothL1Loss()
        elif config.criterions[k] == "l1":
            criterion = F.l1_loss
        elif config.criterions[k] == 'l2':
            criterion = F.mse_loss
        elif config.criterions[k] == "STARLoss":
            criterion = STARLoss(dist=config.star_dist, w=config.star_w)
        elif config.criterions[k] == "STARLoss_v2":
            criterion = STARLoss_v2(dist=config.star_dist, w=config.star_w)
        else:
            assert False
        criterions.append(criterion)
    return criterions


def set_environment(config):
    if config.device_id >= 0:
        if not torch.cuda.is_available() or torch.cuda.device_count() <= config.device_id:
            raise RuntimeError("CUDA device %s is unavailable; use --device_ids=-1 for CPU" % config.device_id)
        torch.cuda.empty_cache()
        config.device = torch.device("cuda", config.device_id)
        config.use_gpu = True
    else:
        config.device = torch.device("cpu")
        config.use_gpu = False

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_flush_denormal(True)  # ignore extremely small value
    torch.backends.cudnn.benchmark = True  # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.autograd.set_detect_anomaly(True)


def forward(config, test_loader, net):
    # ave_metrics = [[0, 0] for i in range(config.label_num)]
    list_nmes = [[] for i in range(config.label_num)]
    metric_nme = NME(nme_left_index=config.nme_left_index, nme_right_index=config.nme_right_index)
    metric_fr_auc = FR_AUC(data_definition=config.data_definition)

    output_pd = None

    net = net.float().to(config.device)
    net.eval()
    dataset_size = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    if config.logger is not None:
        config.logger.info("Forward process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size))
    for i, sample in enumerate(tqdm(test_loader)):
        input = sample["data"].float().to(config.device, non_blocking=True)
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels

        with torch.no_grad():
            output, heatmap, landmarks = net(input)

            # metrics
        for k in range(config.label_num):
            if config.metrics[k] is not None:
                list_nmes[k] += metric_nme.test(output[k], labels[k])

    metrics = [[np.mean(nmes), ] + metric_fr_auc.test(nmes) for nmes in list_nmes]

    return output_pd, metrics


def forward_test(config, test_loader, net):
    # ave_metrics = [[0, 0] for i in range(config.label_num)]
    out_lmd_img_path = config.ckpt_dir+config.data_definition+'/image_results/'
    os.makedirs(out_lmd_img_path, exist_ok=True)

    list_nmes = [[] for i in range(config.label_num)]
    metric_nme = NME(nme_left_index=config.nme_left_index, nme_right_index=config.nme_right_index)
    metric_fr_auc = FR_AUC(data_definition=config.data_definition)

    output_pd = None

    net = net.float().to(config.device)
    net.eval()
    dataset_size = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    if config.logger is not None:
        config.logger.info("Forward process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size))
    for i, sample in enumerate(tqdm(test_loader)):
        input = sample["data"].float().to(config.device, non_blocking=True)
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels

        with torch.no_grad():
            output, heatmap, landmarks = net(input)

            # metrics
        for k in range(config.label_num):
            if config.metrics[k] is not None:
                list_nmes[k] += metric_nme.test(output[k], labels[k])

        # plot image with landmark
        # out_lmd_img_path = os.path.join(
        #     args.res_folder, basename+'_lms.jpg')
       # image = cv2.imread(sample['image_path'][0])
        image = (sample['data'][0].permute(1,2,0).cpu().numpy()+1)*255/2
        style_name = sample['image_path'][0].split('/')[-2]
        base_name = os.path.basename(sample['image_path'][0])
        save_dir = out_lmd_img_path + style_name + base_name
       # landmarks = sample['label'][0]
        landmarks = (landmarks[0]+1)*255/2
        gt_label = (labels[0][0]+1)*255/2
        plot_image = plot_verts(image, gt_label, color='b')
        plot_image = plot_verts(plot_image, landmarks, color='g')
        cv2.imwrite(save_dir, plot_image)
        print()

    metrics = [[np.mean(nmes), ] + metric_fr_auc.test(nmes) for nmes in list_nmes]

    return output_pd, metrics


def _image_output_paths(output_dir, dataset, image_path, source_paths):
    source = Path(image_path).resolve()
    try:
        relative = source.relative_to(Path(dataset.image_dir).resolve())
    except ValueError:
        relative = Path(source.name)
    visualization = output_dir / relative
    landmarks = visualization.with_name(visualization.name + '.npy')
    for target in (visualization, landmarks):
        try:
            target.resolve().relative_to(output_dir)
        except ValueError:
            raise ValueError("Output path leaves --output_dir: %s" % target)
    # Never overwrite an input, including a different image in the same batch.
    if visualization.resolve() in source_paths or landmarks.resolve() in source_paths:
        raise ValueError("Output would overwrite an input image; choose another --output_dir")
    visualization.parent.mkdir(parents=True, exist_ok=True)
    return visualization, landmarks


def forward_test_image_only(config, test_loader, net):
    output_dir = getattr(config, 'output_dir', None)
    if not output_dir:
        output_dir = osp.join(config.ckpt_dir, config.data_definition, 'image_results')
    output_dir = Path(output_dir).expanduser().resolve()
    source_paths = {Path(path).resolve() for path in test_loader.dataset.image_paths}
    if not len(test_loader.dataset):
        raise ValueError("No images to process")

    net = net.float().to(config.device)
    net.eval()

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    total_inference_time = 0
    results = []

    for i, sample in enumerate(tqdm(test_loader)):
        input = sample["data"].float().to(config.device, non_blocking=True)

        with torch.no_grad():
            if config.device.type == 'cuda':
                torch.cuda.synchronize(config.device)
            start_time = time.time()
            output, heatmap, landmarks = net(input)
            if config.device.type == 'cuda':
                torch.cuda.synchronize(config.device)
            end_time = time.time()

        batch_inference_time = end_time - start_time
        total_inference_time += batch_inference_time

        for index, image_path in enumerate(sample['image_path']):
            visualization, landmarks_path = _image_output_paths(
                output_dir, test_loader.dataset, image_path, source_paths)
            image = (sample['data'][index].permute(1, 2, 0).cpu().numpy() + 1) * 255 / 2
            image = np.clip(image, 0, 255).astype(np.uint8)
            # Keep the original model's align-corners mapping. Coordinates are
            # in the resized network image, not the input image's dimensions.
            points = (landmarks[index].detach().cpu().numpy() + 1) * np.array(
                [config.width - 1, config.height - 1], dtype=np.float32) / 2
            if not np.isfinite(points).all():
                raise ValueError("Non-finite landmark prediction for: %s" % image_path)
            plot_image = plot_verts(image, points, color='r')
            if not cv2.imwrite(str(visualization), plot_image):
                raise OSError("Cannot write visualization: %s" % visualization)
            np.save(str(landmarks_path), points)
            results.append({
                'image_path': str(Path(image_path).resolve()),
                'visualization': str(visualization),
                'landmarks': str(landmarks_path),
            })

    average_inference_time = total_inference_time / len(results)
    print(f"Average inference time: {average_inference_time} seconds")
    print("Saved %d image(s) to %s" % (len(results), output_dir))

    return results

def compute_loss(config, criterions, output, labels, heatmap=None, landmarks=None):
    batch_weight = 1.0
    sum_loss = 0
    losses = list()
    for k in range(config.label_num):
        if config.criterions[k] in ['smoothl1', 'l1', 'l2', 'WingLoss', 'AWingLoss']:
            loss = criterions[k](output[k], labels[k])
        elif config.criterions[k] in ["STARLoss", "STARLoss_v2"]:
            _k = int(k / 3) if config.use_AAM else k
            loss = criterions[k](heatmap[_k], labels[k])
        else:
            assert NotImplementedError
        loss = batch_weight * loss
        sum_loss += config.loss_weights[k] * loss
        loss = float(loss.data.cpu().item())
        losses.append(loss)
    return losses, sum_loss



def forward_backward(config, train_loader, net_module, net, net_ema, criterions, optimizer, epoch):
    train_model_time = AverageMeter()
    ave_losses = [0] * config.label_num

    net_module = net_module.float().to(config.device)
    net_module.train(True)
    dataset_size = len(train_loader.dataset)
    batch_size = config.batch_size  # train_loader.batch_size
    batch_num = max(dataset_size / max(batch_size, 1), 1)
    if config.logger is not None:
        config.logger.info(config.note)
        config.logger.info("Forward Backward process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size))

    iter_num = len(train_loader)
    epoch_start_time = time.time()
    if net_module != net:
        train_loader.sampler.set_epoch(epoch)
    for iter, sample in enumerate(train_loader):
        iter_start_time = time.time()
        # input
        input = sample["data"].float().to(config.device, non_blocking=True)
        # labels
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels
        # forward
        output, heatmaps, landmarks = net_module(input)

        # loss
        losses, sum_loss = compute_loss(config, criterions, output, labels, heatmaps, landmarks)
        ave_losses = list(map(sum, zip(ave_losses, losses)))

        # backward
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            sum_loss.backward()
        # torch.nn.utils.clip_grad_norm_(net_module.parameters(), 128.0)
        optimizer.step()

        if net_ema is not None:
            accumulate_net(net_ema, net, 0.5 ** (config.batch_size / 10000.0))
            # accumulate_net(net_ema, net, 0.5 ** (8 / 10000.0))

        # output
        train_model_time.update(time.time() - iter_start_time)
        last_time = convert_secs2time(train_model_time.avg * (iter_num - iter - 1), True)
        if iter % config.display_iteration == 0 or iter + 1 == len(train_loader):
            if config.logger is not None:
                losses_str = ' Average Loss: {:.6f}'.format(sum(losses) / len(losses))
                for k, loss in enumerate(losses):
                    losses_str += ', L{}: {:.3f}'.format(k, loss)
                config.logger.info(
                    ' -->>[{:03d}/{:03d}][{:03d}/{:03d}]'.format(epoch, config.max_epoch, iter, iter_num) \
                    + last_time + losses_str)

        if iter ==0:
            # plot image

            out_lmd_img_path = config.ckpt_dir + config.data_definition + '/image_log/'
            os.makedirs(out_lmd_img_path, exist_ok=True)

            image = (input[0].permute(1, 2, 0).cpu().numpy() + 1) * 255 / 2
            base_name = os.path.basename(sample['image_path'][0])
            save_dir = out_lmd_img_path + str(epoch)+'.jpg'
            lms = (landmarks[0] + 1) * 255 / 2
            plot_image = plot_verts(image, lms, color='r')
            cv2.imwrite(save_dir, plot_image)

            gt_lms = (sample['label'][0][0] + 1) * 255 / 2
            plot_image_ = plot_verts(image, gt_lms, color='b')
            save_dir_ = out_lmd_img_path + str(epoch) + '_gt.jpg'
            cv2.imwrite(save_dir_, plot_image_)

    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time - epoch_start_time
    epoch_load_data_time = epoch_total_time - train_model_time.sum
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average total time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_total_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average loading data time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_load_data_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average training model time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, train_model_time.avg))

    ave_losses = [loss / iter_num for loss in ave_losses]



    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average Loss in this epoch: %.6f" % (
            epoch, config.max_epoch, sum(ave_losses) / len(ave_losses)))
    for k, ave_loss in enumerate(ave_losses):
        if config.logger is not None:
            config.logger.info("Train/Loss%03d in this epoch: %.6f" % (k, ave_loss))


def accumulate_net(model1, model2, decay):
    """
        operation: model1 = model1 * decay + model2 * (1 - decay)
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(
            other=par2[k].data.to(par1[k].data.device),
            alpha=1 - decay)

    par1 = dict(model1.named_buffers())
    par2 = dict(model2.named_buffers())
    for k in par1.keys():
        if par1[k].data.is_floating_point():
            par1[k].data.mul_(decay).add_(
                other=par2[k].data.to(par1[k].data.device),
                alpha=1 - decay)
        else:
            par1[k].data = par2[k].data.to(par1[k].data.device)


def save_model(config, epoch, net, net_ema, optimizer, scheduler, pytorch_model_path):
    # save pytorch model
    state = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }
    if config.ema:
        state["net_ema"] = net_ema.state_dict()

    torch.save(state, pytorch_model_path)
    if config.logger is not None:
        config.logger.info("Epoch: %d/%d, model saved in this epoch" % (epoch, config.max_epoch))

def plot_verts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image,(int(st[0]), int(st[1])), 3, c, -1)

    return image
