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
from utils import ramps, losses
from networks.vnet import VNet_mct
from utils.losses import dice_loss
from dataloaders.CTdataset import CTdataset, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import torch.nn as nn
from torch.nn.parameter import Parameter

torch.set_num_threads(4)

class Linear_vector(nn.Module):
    def __init__(self, n_dim):
        super(Linear_vector, self).__init__()
        self.n_dim = n_dim
        self.paras = Parameter(torch.Tensor(self.n_dim, self.n_dim*2))   # linear coefficients
        self.init_ratio = 1e-3   
        self.initialize()   
    
    def initialize(self):
        for param in self.paras:
            param.data.normal_(0, self.init_ratio)
    
    def forward(self, x):
        # resutl = paras*x   (16*9)=(16*16)*(16*9)
        result = torch.mm(self.paras, x)
        return result

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/Pancreas-CT/data_new_norm/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MCCauSSL', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=5000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

# steps
parser.add_argument('--max_step', type=float,
                    default=60, help='consistency_rampup')  
parser.add_argument('--min_step', type=float,
                    default=60, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "/semi/model/" + args.exp + "/"

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

patch_size = (96, 96, 96)
num_classes = 2

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet_mct(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()

    linear_paras1 = []   
    linear_paras2 = []  
    linear_paras3 = []
    count = 0
    for name, parameters in net.named_parameters():
        if ('block_six' in name or 'block_seven' in name or 'block_eight' in name or 'block_nine' in name or 'out_conv' in name) and 'up' not in name and 'trilinear' not in name and 'nearest' not in name:
            if 'conv' in name and 'weight' in name :
                if len(parameters.shape) == 5:
                    print(name)
                    count += 1
                    outdim = parameters.shape[0] # output dimension
                    linear_paras1.append(Linear_vector(outdim))
                    linear_paras2.append(Linear_vector(outdim))
                    linear_paras3.append(Linear_vector(outdim))

    linear_paras1 = nn.ModuleList(linear_paras1)
    linear_paras2 = nn.ModuleList(linear_paras2)
    linear_paras3 = nn.ModuleList(linear_paras3)
    linear_paras1 = linear_paras1.cuda()
    linear_paras2 = linear_paras2.cuda()
    linear_paras3 = linear_paras3.cuda()
    linear_optimizer1 = torch.optim.Adam(linear_paras1.parameters(), 2e-2)
    linear_optimizer2 = torch.optim.Adam(linear_paras2.parameters(), 2e-2)  
    linear_optimizer3 = torch.optim.Adam(linear_paras3.parameters(), 2e-2)  

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

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    iter_num_max = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            if iter_num > 40 and iter_num%args.min_step == 0:
                for i_max in range(args.max_step):
                    # optimize linear coefficients
                    icm_loss1 = -losses.l_correlation_cos_mean_mct1(net, linear_paras1)
                    icm_loss2 = -losses.l_correlation_cos_mean_mct2(net, linear_paras2)
                    icm_loss3 = -losses.l_correlation_cos_mean_mct3(net, linear_paras3)

                    linear_optimizer1.zero_grad()
                    linear_optimizer2.zero_grad()
                    linear_optimizer3.zero_grad()

                    icm_loss1.backward()
                    icm_loss2.backward()
                    icm_loss3.backward()

                    linear_optimizer1.step()
                    linear_optimizer2.step()
                    linear_optimizer3.step()

                    iter_num_max += 1

                    writer.add_scalar('loss/icm_loss1_max', -icm_loss1, iter_num_max)
                    writer.add_scalar('loss/icm_loss2_max', -icm_loss2, iter_num_max)
                    writer.add_scalar('loss/icm_loss3_max', -icm_loss3, iter_num_max)


            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs1, outputs2, outputs3 = net(volume_batch)

            loss_seg1 = F.cross_entropy(outputs1[:labeled_bs], label_batch[:labeled_bs])
            loss_seg2 = F.cross_entropy(outputs2[:labeled_bs], label_batch[:labeled_bs])
            loss_seg3 = F.cross_entropy(outputs3[:labeled_bs], label_batch[:labeled_bs])

            outputs_soft1 = F.softmax(outputs1, dim=1)
            outputs_soft2 = F.softmax(outputs2, dim=1)
            outputs_soft3 = F.softmax(outputs3, dim=1)

            loss_seg_dice1 = losses.dice_loss(outputs_soft1[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice2 = losses.dice_loss(outputs_soft2[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice3 = losses.dice_loss(outputs_soft3[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            supervised_loss1 = 0.5*(loss_seg1+loss_seg_dice1)
            supervised_loss2 = 0.5*(loss_seg2+loss_seg_dice2)
            supervised_loss3 = 0.5*(loss_seg3+loss_seg_dice3)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            # calculate loss
            T = 0.5
            # 1
            outputs_soft1_sharp = outputs_soft1.pow(1/T)
            outputs_soft1_sharp = outputs_soft1_sharp/(torch.sum(outputs_soft1_sharp, dim = 1).unsqueeze(1))
            outputs_soft1_sharp = outputs_soft1_sharp.detach()
            # 2
            outputs_soft2_sharp = outputs_soft2.pow(1/T)
            outputs_soft2_sharp = outputs_soft2_sharp/(torch.sum(outputs_soft2_sharp, dim = 1).unsqueeze(1))
            outputs_soft2_sharp = outputs_soft2_sharp.detach()
            # 3
            outputs_soft3_sharp = outputs_soft3.pow(1/T)
            outputs_soft3_sharp = outputs_soft3_sharp/(torch.sum(outputs_soft3_sharp, dim = 1).unsqueeze(1))
            outputs_soft3_sharp = outputs_soft3_sharp.detach()
            # 
            consistency_loss1 = torch.mean((outputs_soft1[args.labeled_bs:]-outputs_soft2_sharp[args.labeled_bs:])**2) + torch.mean((outputs_soft1[args.labeled_bs:]-outputs_soft3_sharp[args.labeled_bs:])**2)
            consistency_loss2 = torch.mean((outputs_soft2[args.labeled_bs:]-outputs_soft1_sharp[args.labeled_bs:])**2) + torch.mean((outputs_soft2[args.labeled_bs:]-outputs_soft3_sharp[args.labeled_bs:])**2)
            consistency_loss3 = torch.mean((outputs_soft3[args.labeled_bs:]-outputs_soft1_sharp[args.labeled_bs:])**2) + torch.mean((outputs_soft3[args.labeled_bs:]-outputs_soft2_sharp[args.labeled_bs:])**2)
            consistency_dist = consistency_loss1 + consistency_loss2 + consistency_loss3
            

            if iter_num > 40 and iter_num_max > 0:
                icm_loss1 = losses.l_correlation_cos_mean_mct1(net, linear_paras1)
                icm_loss2 = losses.l_correlation_cos_mean_mct2(net, linear_paras2)
                icm_loss3 = losses.l_correlation_cos_mean_mct3(net, linear_paras3)
            else:
                icm_loss1 = 0
                icm_loss2 = 0
                icm_loss3 = 0


            loss = supervised_loss1 + supervised_loss2 + supervised_loss3 + consistency_weight * consistency_dist + 3.0*(icm_loss1 + icm_loss2 + icm_loss3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg1', loss_seg1, iter_num)
            writer.add_scalar('loss/loss_seg2', loss_seg2, iter_num)
            writer.add_scalar('loss/loss_seg3', loss_seg3, iter_num)
            writer.add_scalar('loss/loss_seg_dice1', loss_seg_dice1, iter_num)
            writer.add_scalar('loss/loss_seg_dice2', loss_seg_dice2, iter_num)
            writer.add_scalar('loss/loss_seg_dice3', loss_seg_dice3, iter_num)
            writer.add_scalar('loss/consistency_loss1', consistency_loss1, iter_num)
            writer.add_scalar('loss/consistency_loss2', consistency_loss2, iter_num)
            writer.add_scalar('loss/consistency_loss3', consistency_loss3, iter_num)
            writer.add_scalar('loss/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('loss/icm_loss1_min', icm_loss1, iter_num)
            writer.add_scalar('loss/icm_loss2_min', icm_loss2, iter_num)
            writer.add_scalar('loss/icm_loss3_min', icm_loss3, iter_num)

            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
