import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    domain_labels = input_list[2]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    # batch_size = softmax_output.size(0) // 2
    # dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    dc_target = domain_labels.unsqueeze(-1).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        # source_mask[feature.size(0)//2:] = 0
        source_mask[np.where(domain_labels.cpu().numpy() == 0)[0].tolist()] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        # target_mask[0:feature.size(0)//2] = 0
        target_mask[np.where(domain_labels.cpu().numpy() == 1)[0].tolist()] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        ####################################################################
        # Changed @ 2023/11/7
        if max(dc_target) <= 1:
            return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
        else:
            return torch.sum(weight.view(-1, 1) * nn.CrossEntropyLoss(reduction='none')(ad_out, dc_target))  / torch.sum(weight).detach().item()
        ####################################################################
    else:
        ####################################################################
        # Changed @ 2023/11/7
        if max(dc_target) <= 1:
            return nn.BCELoss()(ad_out, dc_target) 
        else:
            dc_target = dc_target / max(dc_target)
            return nn.CrossEntropyLoss()(ad_out, dc_target) 
        ####################################################################


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)
