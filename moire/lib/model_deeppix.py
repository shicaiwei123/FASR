
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
from torchvision.datasets import ImageFolder
from lib.model_utils import GradualWarmupScheduler

def one_hot(label_list):
    count = 0

    for label in label_list:
        if label == 1:
            if count == 0:
                label_new = torch.tensor([[0, 1]])
            else:
                label_tensor = torch.tensor([[0, 1]])
        else:
            if count == 0:
                label_new = torch.tensor([[1, 0]])
            else:
                label_tensor = torch.tensor([[1, 0]])
        if count >= 1:
            label_new = torch.cat((label_new, label_tensor))
        count += 1
    return label_new


def calc_accuracy(model, loader, verbose=False):
    # """
    # :param model: model network
    # :param loader: torch.utils.data.DataLoader
    # :param verbose: show progress bar, bool
    # :return accuracy, float
    # """
    # use_cuda = torch.cuda.is_available()
    # if use_cuda:
    #     model.cuda()
    # outputs_full = []
    # labels_full = []
    #
    # # for idx, batch_sample in enumerate(
    # #         tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose)):
    #
    # for idx, batch_sample in enumerate(loader):
    #     inputs, map_target, target = batch_sample['image_x'], batch_sample['map_x'], \
    #                                  batch_sample['spoofing_label']
    #
    #     use_cuda = torch.cuda.is_available()
    #     if use_cuda:
    #         model.cuda()
    #     outputs_full = []
    #     labels_full = []
    #
    #     if torch.cuda.is_available():
    #         inputs, map_target, target = inputs.cuda(), map_target.cuda(), target.cuda()
    #
    #     with torch.no_grad():
    #         map_out, label_out = model(inputs)
    #     outputs_full.append(label_out)
    #     labels_full.append(target)
    #
    # outputs_full = torch.cat(outputs_full, dim=0)
    # labels_full = torch.cat(labels_full, dim=0)
    # _, labels_predicted = torch.max(outputs_full.data, dim=1)
    # accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # return accuracy

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for batch_sample in (loader):

        inputs, map_target, labels = batch_sample['image_x'], batch_sample['map_x'], \
                                     batch_sample['spoofing_label']

        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            map_out, outputs_batch = model(inputs)
        outputs_full.append(outputs_batch)
        labels_full.append(labels)
    # model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    return accuracy


def train_base(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model: ??????????????????
    :param cost: ????????????
    :param optimizer: ?????????
    :param train_loader:  ??????????????????
    :param test_loader:  ??????????????????
    :param args: ????????????
    :return:
    '''
    criterion_absolute_loss = nn.BCELoss()
    # ??????????????????
    print(args)

    # ?????????,???????????????,??????????????????
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name +'.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # ??????argv??????
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # ??????????????????????????????
    if args.lrcos:
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=0)

        # ???????????????????????????
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # ???????????????
    epoch_num = args.train_epoch  # ???????????????epoch
    log_interval = args.log_interval  # ???????????????batch??????????????????
    save_interval = args.save_interval  # ???????????????epoch ??????????????????

    batch_num = 0
    train_loss = 0
    log_list = []  # ???????????????log????????????

    epoch = 0
    accuracy_best = 0

    # ??????????????????????????????,????????????????????????????????????
    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # ??????
    while epoch < epoch_num:  # ???epoch?????????????????????
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            data, map_target, target = batch_sample['image_x'], batch_sample['map_x'], \
                                       batch_sample['spoofing_label']

            map_target = map_target.unsqueeze(1)

            batch_num += 1
            if torch.cuda.is_available():
                data, map_target, target = data.cuda(), map_target.cuda(), target.cuda()

            optimizer.zero_grad()  # ??????????????????????????????,??????????????????????????????

            map_out, label_out = model(data)  # ????????????????????????????????????????????????????????????

            target = one_hot(target)
            if torch.cuda.is_available():
                target = target.cuda()
            loss1 = cost(label_out, target.float())

            loss2 = criterion_absolute_loss(map_out, map_target.float())

            loss = (loss1 + loss2) / 2

            train_loss += loss.item()
            loss.backward()  # ???????????????????????????????????????????????????
            optimizer.step()  # ???????????????????????????????????????,??????????????????```````````````````````````````````````````````````
            # if batch_idx % log_interval == 0:  # ???????????????????????????args.log_interval????????????????????????????????????
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # ????????????
        model = model.eval()
        accuracy_test = calc_accuracy(model, loader=test_loader)
        model = model.train()

        if epoch > 5:
            if accuracy_test > accuracy_best:
                accuracy_best = accuracy_test
                torch.save(model.state_dict(), args.name+".pth")
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lrcos:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # ??????????????????????????????
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            torch.save(train_state, models_dir)

        # ??????log
        with open(log_dir, 'a+', newline='') as f:
            # ????????????
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)




def deploy_base(model, img, transform):
    '''
    ??????????????????????????????????????????
    :param model: ??????
    :param img: PIL.Image?????????
    :transform ???????????????????????????,???test?????????
    :return:
    '''

    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        img_tensor = img_tensor.cuda()
    map_result, label_result = model(img_tensor)

    if use_cuda:
        label_result = label_result.cpu()
    result = label_result.detach().numpy()
    result = result[0]

    return result
