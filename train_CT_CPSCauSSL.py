import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.CTdataset import CTdataset, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
from test_util_CT import test_all_case
# device
import os
import pandas as pd

torch.set_num_threads(4)

class Linear_vector(nn.Module):
    def __init__(self, n_dim):
        super(Linear_vector, self).__init__()
        self.n_dim = n_dim
        self.paras = Parameter(torch.Tensor(self.n_dim, self.n_dim))   # linear coefficients
        self.init_ratio = 1e-3   
        self.initialize()   
    
    def initialize(self):
        for param in self.paras:
            param.data.normal_(0, self.init_ratio)
    
    def forward(self, x):
        # result = paras*x   (16*9)=(16*16)*(16*9)
        result = torch.mm(self.paras, x)
        return result


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/Pancreas-CT/data_new_norm/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='CPSCauSS', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=5000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

# steps
parser.add_argument('--max_step', type=float,
                    default=60, help='consistency_rampup') 
parser.add_argument('--min_step', type=float,
                    default=60, help='consistency_rampup')
parser.add_argument('--start_step1', type=float,
                    default=500, help='consistency_rampup')
parser.add_argument('--start_step2', type=float,
                    default=500, help='consistency_rampup')
parser.add_argument('--coefficient', type=float,
                    default=3.0, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path 
snapshot_path = "/semi/model/" + args.exp + "_max_" + str(args.max_step) + "_min_" + str(args.min_step) + "_start1_" + str(args.start_step1) + "_start2_" + str(args.start_step2) + "_coe_" + str(args.coefficient) + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()

    db_train = CTdataset(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = CTdataset(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(12))
    unlabeled_idxs = list(range(12, 62))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    linear_paras1 = [] 
    linear_paras2 = []  
    count = 0
    for name, parameters in model1.named_parameters():
        if 'conv' in name and 'weight' in name :
            if len(parameters.shape) == 5:
                count += 1
                outdim = parameters.shape[0] # output dimension
                linear_paras1.append(Linear_vector(outdim))
                linear_paras2.append(Linear_vector(outdim))

    linear_paras1 = nn.ModuleList(linear_paras1)
    linear_paras2 = nn.ModuleList(linear_paras2)
    linear_paras1 = linear_paras1.cuda()
    linear_paras2 = linear_paras2.cuda()
    linear_optimizer1 = torch.optim.Adam(linear_paras1.parameters(), 2e-2)
    linear_optimizer2 = torch.optim.Adam(linear_paras2.parameters(), 2e-2)    

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    iter_num_max = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model1.train()
    model2.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            if iter_num > args.start_step1 and iter_num%args.min_step == 0:
                for i_max in range(args.max_step):
                    # optimize linear coefficients
                    icm_loss1 = -losses.l_correlation_cos_mean(model1, model2, linear_paras1)
                    icm_loss2 = -losses.l_correlation_cos_mean(model2, model1, linear_paras2)

                    linear_optimizer1.zero_grad()
                    linear_optimizer2.zero_grad()

                    icm_loss1.backward()
                    icm_loss2.backward()

                    linear_optimizer1.step()
                    linear_optimizer2.step()

                    iter_num_max += 1

                    writer.add_scalar('loss/icm_loss1_max', -icm_loss1, iter_num_max)
                    writer.add_scalar('loss/icm_loss2_max', -icm_loss2, iter_num_max)

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            outputs1 = model1(volume_batch)
            outputs2 = model2(volume_batch)

            ## calculate the loss
            loss_seg1 = F.cross_entropy(outputs1[:labeled_bs], label_batch[:labeled_bs])
            loss_seg2 = F.cross_entropy(outputs2[:labeled_bs], label_batch[:labeled_bs])

            outputs_soft1 = F.softmax(outputs1, dim=1)
            outputs_soft2 = F.softmax(outputs2, dim=1)

            loss_seg_dice1 = losses.dice_loss(outputs_soft1[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice2 = losses.dice_loss(outputs_soft2[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            supervised_loss1 = 0.5*(loss_seg1+loss_seg_dice1)
            supervised_loss2 = 0.5*(loss_seg2+loss_seg_dice2)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            if iter_num > 400:
                consistency_dist1 = F.cross_entropy(outputs1[args.labeled_bs:], pseudo_outputs2)
                consistency_dist2 = F.cross_entropy(outputs2[args.labeled_bs:], pseudo_outputs1)
            else:
                consistency_dist1 = 0
                consistency_dist2 = 0

            consistency_loss1 = consistency_weight * 0.3*consistency_dist1
            consistency_loss2 = consistency_weight * 0.3*consistency_dist2

            if iter_num > args.start_step2 and iter_num_max > 0:
                icm_loss1 = losses.l_correlation_cos_mean(model1, model2, linear_paras1)
                icm_loss2 = losses.l_correlation_cos_mean(model2, model1, linear_paras2)
            else:
                icm_loss1 = 0
                icm_loss2 = 0

            loss1 = supervised_loss1 + consistency_loss1 + args.coefficient*icm_loss1
            loss2 = supervised_loss2 + consistency_loss2 + args.coefficient*icm_loss2

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss1', loss1, iter_num)
            writer.add_scalar('loss/loss_seg1', loss_seg1, iter_num)
            writer.add_scalar('loss/loss_seg_dice1', loss_seg_dice1, iter_num)
            writer.add_scalar('train/consistency_loss1', consistency_loss1, iter_num)
            writer.add_scalar('train/consistency_dist1', consistency_dist1, iter_num)
            writer.add_scalar('loss/loss2', loss2, iter_num)
            writer.add_scalar('loss/loss_seg2', loss_seg2, iter_num)
            writer.add_scalar('loss/loss_seg_dice2', loss_seg_dice2, iter_num)
            writer.add_scalar('train/consistency_loss2', consistency_loss2, iter_num)
            writer.add_scalar('train/consistency_dist2', consistency_dist2, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)

            writer.add_scalar('loss/icm_loss1_min', icm_loss1, iter_num)
            writer.add_scalar('loss/icm_loss2_min', icm_loss2, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter1_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                save_mode_path = os.path.join(snapshot_path, 'iter2_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter1_'+str(max_iterations)+'.pth')
    torch.save(model1.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    save_mode_path = os.path.join(snapshot_path, 'iter2_'+str(max_iterations)+'.pth')
    torch.save(model2.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

    # testing
    with open('/Pancreas-CT/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path +item.replace('\n', '')+".h5" for item in image_list]


    def test_calculate_metric_1(epoch_num):
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
        save_mode_path = os.path.join(snapshot_path, 'iter1_' + str(epoch_num) + '.pth')
        net.load_state_dict(torch.load(save_mode_path))
        print("init weight from {}".format(save_mode_path))
        net.eval()

        avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                                patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                save_result=False)
        return avg_metric
    
    def test_calculate_metric_2(epoch_num):
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
        save_mode_path = os.path.join(snapshot_path, 'iter2_' + str(epoch_num) + '.pth')
        net.load_state_dict(torch.load(save_mode_path))
        print("init weight from {}".format(save_mode_path))
        net.eval()

        avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                                patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                save_result=False)

        return avg_metric

    nums = [5000]
    first_list = np.zeros([1, 4])
    second_list = np.zeros([1, 4])
    count = 0
    for i in nums:
        metric1 = test_calculate_metric_1(i)
        first_list[count, :] = metric1
        metric2 = test_calculate_metric_2(i)

        second_list[count, :] = metric2
        count += 1

    write_csv = "/test_csv/" + args.exp + "_max_" + str(args.max_step) + "_min_" + str(args.min_step) + "_start1_" + str(args.start_step1) + "_start2_" + str(args.start_step2) + "_coe_" + str(args.coefficient) +  "_1.csv"
    save = pd.DataFrame({'dice':first_list[:,0], 'jc':first_list[:,1], 'hd95':first_list[:,2], 'asd':first_list[:,3]})
    save.to_csv(write_csv, index=False, sep=',')

    write_csv = "/test_csv/" + args.exp + "_max_" + str(args.max_step) + "_min_" + str(args.min_step) + "_start1_" + str(args.start_step1) + "_start2_" + str(args.start_step2) + "_coe_" + str(args.coefficient) + "_2.csv"
    save = pd.DataFrame({'dice':second_list[:,0], 'jc':second_list[:,1], 'hd95':second_list[:,2], 'asd':second_list[:,3]})
    save.to_csv(write_csv, index=False, sep=',')
