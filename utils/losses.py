import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

def l_correlation_cos_mean(model1, model2, linear_paras1):    # fix model2
    total_loss = 0.0
    count = 0
    for name, parameters in model1.named_parameters():
        if 'conv' in name and 'weight' in name :
            if len(parameters.shape) == 5:
                w1 = parameters
                w2 = model2.state_dict()[name]
                w2 = w2.detach()
                outdim = parameters.shape[0]
                w1 = w1.view(outdim,-1)
                w2 = w2.view(outdim,-1)
                out = linear_paras1[count](w2)

                # normalization
                out = nn.functional.normalize(out, dim=1)  # N*C
                w1_d = nn.functional.normalize(w1, dim=1)

                loss_temp = torch.einsum('nc, nc->n', [out, w1_d])   # x*p
                total_loss += torch.mean(loss_temp*loss_temp)


                count += 1
    
    total_loss = total_loss/count   
    return total_loss

def l_correlation_cos_mean_mct1(model1, linear_paras1):    # first branch
    total_loss = 0.0
    count = 0
    for name, parameters in model1.named_parameters():
        if ('block_six' in name or 'block_seven' in name or 'block_eight' in name or 'block_nine' in name or 'out_conv' in name) and 'up' not in name and 'trilinear' not in name and 'nearest' not in name:
            if 'conv' in name and 'weight' in name:
                if len(parameters.shape) == 5:
                    w1 = parameters
                    name_temp = name.split('.')
                    name2 = name_temp[0]+'_trilinear'
                    for i in range(1, len(name_temp)):
                        name2 = name2 + '.' + name_temp[i]
                    name3 = name_temp[0]+'_nearest'
                    for i in range(1, len(name_temp)):
                        name3 = name3 + '.' + name_temp[i]
                    w2 = model1.state_dict()[name2]
                    w2 = w2.detach()
                    w3 = model1.state_dict()[name3]
                    w3 = w3.detach()

                    outdim = parameters.shape[0]
                    w1 = w1.view(outdim,-1)
                    w2 = w2.view(outdim,-1)
                    w3 = w3.view(outdim,-1)
                    w_new = torch.cat([w2, w3], dim = 0)
                    out = linear_paras1[count](w_new)

                    # normalization
                    out = nn.functional.normalize(out, dim=1)  # N*C
                    w1_d = nn.functional.normalize(w1, dim=1)

                    loss_temp = torch.einsum('nc, nc->n', [out, w1_d])   # x*p
                    total_loss += torch.mean(loss_temp*loss_temp)

                    count += 1

    total_loss = total_loss/count  
    return total_loss

def l_correlation_cos_mean_mct2(model1, linear_paras1):    # second branch
    total_loss = 0.0
    count = 0
    for name, parameters in model1.named_parameters():
        if ('block_six' in name or 'block_seven' in name or 'block_eight' in name or 'block_nine' in name or 'out_conv' in name) and 'up' not in name and 'trilinear' not in name and 'nearest' not in name:
            if 'conv' in name and 'weight' in name:
                if len(parameters.shape) == 5:
                    w2 = parameters
                    name_temp = name.split('.')
                    name2 = name_temp[0]+'_trilinear'
                    for i in range(1, len(name_temp)):
                        name2 = name2 + '.' + name_temp[i]
                    name3 = name_temp[0]+'_nearest'
                    for i in range(1, len(name_temp)):
                        name3 = name3 + '.' + name_temp[i]
                    w1 = model1.state_dict()[name2]
                    w2 = w2.detach()
                    w3 = model1.state_dict()[name3]
                    w3 = w3.detach()

                    outdim = parameters.shape[0]
                    w1 = w1.view(outdim,-1)
                    w2 = w2.view(outdim,-1)
                    w3 = w3.view(outdim,-1)
                    w_new = torch.cat([w2, w3], dim = 0)
                    out = linear_paras1[count](w_new)

                    # normalization
                    out = nn.functional.normalize(out, dim=1)  # N*C
                    w1_d = nn.functional.normalize(w1, dim=1)

                    loss_temp = torch.einsum('nc, nc->n', [out, w1_d])   # x*p
                    total_loss += torch.mean(loss_temp*loss_temp)

                    count += 1

    total_loss = total_loss/count 
    return total_loss

def l_correlation_cos_mean_mct3(model1, linear_paras1):    # third branch
    total_loss = 0.0
    count = 0
    for name, parameters in model1.named_parameters():
        if ('block_six' in name or 'block_seven' in name or 'block_eight' in name or 'block_nine' in name or 'out_conv' in name) and 'up' not in name and 'trilinear' not in name and 'nearest' not in name:
            if 'conv' in name and 'weight' in name:
                if len(parameters.shape) == 5:
                    w3 = parameters
                    name_temp = name.split('.')
                    name2 = name_temp[0]+'_trilinear'
                    for i in range(1, len(name_temp)):
                        name2 = name2 + '.' + name_temp[i]
                    name3 = name_temp[0]+'_nearest'
                    for i in range(1, len(name_temp)):
                        name3 = name3 + '.' + name_temp[i]
                    w2 = model1.state_dict()[name2]
                    w2 = w2.detach()
                    w1 = model1.state_dict()[name3]
                    w3 = w3.detach()

                    outdim = parameters.shape[0]
                    w1 = w1.view(outdim,-1)
                    w2 = w2.view(outdim,-1)
                    w3 = w3.view(outdim,-1)
                    w_new = torch.cat([w2, w3], dim = 0)
                    out = linear_paras1[count](w_new)

                    # normalization
                    out = nn.functional.normalize(out, dim=1)  # N*C
                    w1_d = nn.functional.normalize(w1, dim=1)

                    loss_temp = torch.einsum('nc, nc->n', [out, w1_d])   # x*p
                    total_loss += torch.mean(loss_temp*loss_temp)

                    count += 1

    total_loss = total_loss/count   
    return total_loss