import argparse
import re
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from tqdm import tqdm
import logging
from myfunc import set_determinism, set_logger


def image_classification_test(loader, model, test_10crop=False):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader[i]) for i in range(10)]
            for i in range(len(loader[0])):
                data = [next(iter_test[j]) for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            for data in tqdm(loader, desc='Testing: '):
                inputs = data[0].cuda()
                labels = data[1]
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def _get_start_epoch(config):
    from glob import glob
    file_names = glob(os.path.join(config["output_path"], 'iter_*.pth.tar'))
    if len(file_names) == 0:
        return 0
    max_iter_file = max(file_names, key=lambda name: int(re.search(r'iter_(\d+)_model.pth.tar', name).group(1)))
    max_iter_number = int(re.search(r'iter_(\d+)_model.pth.tar', max_iter_file).group(1))
    # with open(os.path.join(config["output_path"], 'log.txt'), 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[::-1]:
    #         try:
    #             if 'iter' in line:
    #                 start_epoch = int(re.search(r'iter: (\d+)', line).group(1))
    #                 start_epoch += 1
    #                 break
    #         except:
    #             start_epoch = 0
    return max_iter_number

def train(config):
    logger = logging.getLogger(__name__)
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    start_epoch = 0

    # Resume
    if config['resume']:
        start_epoch = _get_start_epoch(config)
        base_network = torch.load(os.path.join(config["output_path"], f"iter_{start_epoch}_model.pth.tar"))
        if start_epoch >= config["num_iterations"] - 5:
            return
        logger.info(f"Resume from: {start_epoch} epoch\t iter_{start_epoch}_model.pth.tar\n")

    for i in tqdm(range(start_epoch, config["num_iterations"]), desc='Training: '):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders['test'], \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.3f}%".format(i, temp_acc * 100)
            # config["out_file"].write(log_str+"\n")
            # config["out_file"].flush()
            logger.info(log_str)
        if i != 0 and i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]                  
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0 or i == start_epoch:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0 or i == start_epoch:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source, domain_source = next(iter_source)
        inputs_target, labels_target, domain_target = next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        domain_source, domain_target = domain_source.cuda(), domain_target.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        domain_labels = torch.cat((domain_source, domain_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        if config['method'] == 'CDAN+E':           
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out, domain_labels], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out, domain_labels], ad_net, None, None, random_layer)
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dataset', type=str, default='Office31', choices=['Office31', 'image-clef', 'visda', 'OfficeHome', 'DomainNet'], help="The dataset or source dataset used")
    parser.add_argument('--dset', type=str, default='amazon_dslr')
    parser.add_argument('--task', type=str, default='true_domains')
    parser.add_argument('--resume', type=str, default='')  # 'CDAN/Office31/210129_16:00:00--c0123n0--amazon_dslr--true_domains  のように, methodとparetを示す親ディレクトリも書く.
    # parser.add_argument('--s_dset_path', type=str, default='../../data/Office31/amazon_31_list.txt', help="The source dataset path list")
    # parser.add_argument('--t_dset_path', type=str, default='../../data/Office31/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    # parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    
    cuda = ''.join([str(i) for i in os.environ['CUDA_VISIBLE_DEVICES']])
    exec_num = os.environ['exec_num'] if 'exec_num' in os.environ.keys() else 0
    if args.resume:
        assert len(args.resume.split('/')) == 3, 'CDAN/Office31/210129_16:00:00--c0123n0--amazon_dslr--true_domains のように, methodとparetを示す親ディレクトリも書く.'
        args.output_dir = args.resume
        args.method = args.resume.split('/')[0]
        args.dataset = args.resume.split('/')[1]
        args.dset = args.resume.split('--')[2]
        args.task = args.resume.split('--')[3]
        print(f'''
            args.output_dir : \t {args.output_dir}
            args.method : \t {args.method}
            args.dataset : \t {args.dataset}
            args.dset : \t {args.dset}
            args.task : \t {args.task}
        ''')
    else:
        from datetime import datetime
        now = datetime.now().strftime("%y%m%d_%H:%M:%S")
        args.output_dir = f"{args.method}/{args.dataset}/{now}--c{cuda}n{exec_num}--{args.dset}--{args.task}"

    args.s_dset_path = os.path.join('/nas/data/syamagami/GDA/data/GDA_DA_methods/data', args.dataset, args.task, args.dset, 'labeled.txt')
    args.t_dset_path = os.path.join('/nas/data/syamagami/GDA/data/GDA_DA_methods/data', args.dataset, args.task, args.dset, 'unlabeled.txt')

    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 35004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["resume"] = args.resume
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    # if not osp.exists(config["output_path"]):
    #     os.system('mkdir -p '+config["output_path"])
    # config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    os.makedirs(config["output_path"], exist_ok=True)

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dataset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":4}}

    if config["dataset"] == "Office31":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "OfficeHome":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "DomainNet":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 345
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    # config["out_file"].write(str(config))
    # config["out_file"].flush()
    set_determinism()
    set_logger(config['output_path'])
    train(config)
