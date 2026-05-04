import torch
from torch import Tensor, nn
import torchvision
import os
import numpy as np
from resnet_nole import *
from vit_wrapper import ViTWrapper
from models import meta_classifier_cifar10_model,lenet,ULP_model,preact_resnet
import torch.nn.functional as F

import unet_model
import random
import pilgram
from PIL import Image
from functools import reduce


class ConvNetWrapper(nn.Module):
    """
    Wrapper for ConvNet architecture from stealthy_backdoors.
    This matches the architecture used in the backdoored models.
    """
    def __init__(self, input_size=32, input_channels=3, num_classes=10, 
                 kernel_size=5, filters1=64, filters2=64, fc_size=384):
        super(ConvNetWrapper, self).__init__()
        self.input_size = input_size
        self.filters1 = filters1
        self.filters2 = filters2
        
        # Calculate padding to maintain spatial dimensions
        padding = (kernel_size - 1) // 2
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=filters1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=filters1,
            out_channels=filters2,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # After two pooling operations, spatial size becomes (input_size // 4)
        # Fully connected layers
        fc_input_size = (input_size // 4) * (input_size // 4) * filters2
        self.fc1 = nn.Linear(fc_input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, (self.input_size // 4) * (self.input_size // 4) * self.filters2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def from_input_to_features(self, x, index):
        """Extract features after convolutional layers, before FC layers."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Return flattened features
        x = x.view(x.size(0), -1)
        return x
    
    def from_features_to_output(self, x, index):
        """Convert features to output logits."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RegressionModel(nn.Module):
    def __init__(self, opt, init_mask):
        self._EPSILON = opt.EPSILON
        super(RegressionModel, self).__init__()

        if init_mask is not None:
            self.mask_tanh = nn.Parameter(torch.tensor(init_mask))

        self.classifier = self._get_classifier(opt)
        self.example_features = None

        if opt.dataset == "mnist":
            self.AE = unet_model.UNet(n_channels=1,num_classes=1,base_filter_num=opt.ae_filter_num, num_blocks=opt.ae_num_blocks)
        else:
            self.AE = unet_model.UNet(n_channels=3,num_classes=3,base_filter_num=opt.ae_filter_num, num_blocks=opt.ae_num_blocks)

        self.AE.train()
        self.example_ori_img = None
        self.example_ae_img = None
        self.opt = opt

    def forward_ori(self, x,opt):

        features = self.classifier.from_input_to_features(x, opt.internal_index)
        out = self.classifier.from_features_to_output(features, opt.internal_index)

        return out, features

    def forward_flip_mask(self, x,opt):

        strategy = "flip"
        features = self.classifier.from_input_to_features(x, opt.internal_index)
        if strategy == "flip":
            features = (1 - opt.flip_mask) * features - opt.flip_mask * features
        elif strategy == "zero":
            features = (1 - opt.flip_mask) * features

        out = self.classifier.from_features_to_output(features, opt.internal_index)

        return out, features

    def forward_ae(self, x,opt):

        self.example_ori_img = x
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        self.example_ae_img = x

        features = self.classifier.from_input_to_features(x, opt.internal_index)
        out = self.classifier.from_features_to_output(features, opt.internal_index)

        self.example_features = features

        return out, features, x_before_ae, x_after_ae


    def forward_ae_mask_p(self, x,opt):
        mask = self.get_raw_mask(opt)
        self.example_ori_img = x
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        self.example_ae_img = x

        features = self.classifier.from_input_to_features(x, opt.internal_index)
        reference_features_index_list = np.random.choice(range(opt.all_features.shape[0]), features.shape[0], replace=True)
        reference_features = opt.all_features[reference_features_index_list]
        features_ori = features
        features = mask * features + (1-mask) * reference_features.reshape(features.shape)

        out = self.classifier.from_features_to_output(features, opt.internal_index)

        self.example_features = features_ori

        return out, features, x_before_ae, x_after_ae, features_ori

    def forward_ae_mask_p_test(self, x,opt):
        mask = self.get_raw_mask(opt)
        self.example_ori_img = x
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        self.example_ae_img = x

        features = self.classifier.from_input_to_features(x, opt.internal_index)
        bs = features.shape[0]
        index_1 = list(range(bs))
        random.shuffle(index_1)
        reference_features = features[index_1]
        features_ori = features
        features = mask * features + (1-mask) * reference_features.reshape(features.shape)
        out = self.classifier.from_features_to_output(features, opt.internal_index)
        self.example_features = features_ori

        return out, features, x_before_ae, x_after_ae, features_ori

    def get_raw_mask(self,opt):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded

    def _get_classifier(self, opt):

        if opt.hand_set_model_path:
            ckpt_path = opt.hand_set_model_path
            # Load checkpoint to inspect its structure
            state_dict = torch.load(ckpt_path, map_location='cpu')
            
            # Check if this is a direct state_dict (OrderedDict) or wrapped in a dict
            # Check if it's a wrapped checkpoint
            if isinstance(state_dict, dict):
                if "net_state_dict" in state_dict:
                    actual_state_dict = state_dict["net_state_dict"]
                elif "netC" in state_dict:
                    actual_state_dict = state_dict["netC"]
                elif "model" in state_dict:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict["model"].items():
                        name = k[7:] if k.startswith("module.") else k  # remove `module.` prefix
                        new_state_dict[name] = v
                    actual_state_dict = new_state_dict
                elif "model_state_dict" in state_dict:
                    actual_state_dict = state_dict["model_state_dict"]
                else:
                    # Assume it's a direct state_dict (OrderedDict is a subclass of dict)
                    actual_state_dict = state_dict
            else:
                # Direct state_dict (shouldn't happen, but handle it)
                actual_state_dict = state_dict
            
            # Detect architecture from checkpoint keys
            checkpoint_keys = set(actual_state_dict.keys())
            
            # Check if it's a ConvNet (from stealthy_backdoors)
            # ConvNet has: conv1.weight, conv1.bias, conv2.weight, conv2.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias
            if all(key in checkpoint_keys for key in ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"]):
                # It's a ConvNet - infer parameters from checkpoint
                conv1_shape = actual_state_dict["conv1.weight"].shape
                fc1_shape = actual_state_dict["fc1.weight"].shape
                fc2_shape = actual_state_dict["fc2.weight"].shape
                
                # Infer architecture parameters
                filters1 = conv1_shape[0]  # 64
                filters2 = actual_state_dict["conv2.weight"].shape[0]  # 64
                fc_size = fc2_shape[1]  # 384
                num_classes = fc2_shape[0]  # 10
                
                # Infer input size from fc1 input size
                # fc_input_size = (input_size // 4) * (input_size // 4) * filters2
                # So: (input_size // 4)^2 * filters2 = fc1_shape[1]
                fc_input_size = fc1_shape[1]
                spatial_size_squared = fc_input_size // filters2
                spatial_size = int(spatial_size_squared ** 0.5)
                input_size = spatial_size * 4  # Should be 32 for CIFAR10
                
                input_channels = conv1_shape[1]  # 3 for RGB
                
                print(f"Detected ConvNet architecture: input_size={input_size}, input_channels={input_channels}, "
                      f"num_classes={num_classes}, filters1={filters1}, filters2={filters2}, fc_size={fc_size}")
                
                classifier = ConvNetWrapper(
                    input_size=input_size,
                    input_channels=input_channels,
                    num_classes=num_classes,
                    kernel_size=5,  # Default for CIFAR10
                    filters1=filters1,
                    filters2=filters2,
                    fc_size=fc_size
                )
                classifier.load_state_dict(actual_state_dict)
            elif "cls_token" in checkpoint_keys and "patch_embed.proj.weight" in checkpoint_keys and \
                 any(k.startswith("blocks.") for k in checkpoint_keys):
                # ViT from timm — infer params from checkpoint shapes
                import math
                head_shape = actual_state_dict["head.weight"].shape
                num_classes = head_shape[0]
                in_channels = actual_state_dict["patch_embed.proj.weight"].shape[1]
                patch_size = actual_state_dict["patch_embed.proj.weight"].shape[2]
                num_tokens = actual_state_dict["pos_embed"].shape[1]
                grid_size = int(math.sqrt(num_tokens - 1))
                img_size = grid_size * patch_size
                classifier = ViTWrapper(num_classes=num_classes, in_channels=in_channels,
                                        img_size=img_size, patch_size=patch_size)
                classifier.load_state_dict(actual_state_dict)
            else:
                # Use the specified architecture
                if opt.set_arch:
                    if opt.set_arch == "resnet18":
                        classifier = resnet18(num_classes = opt.num_classes, in_channels = opt.input_channel)
                    elif opt.set_arch=="preact_resnet18":
                        classifier = preact_resnet.PreActResNet18(num_classes=opt.num_classes)
                    elif opt.set_arch=="meta_classifier_cifar10_model":
                        classifier = meta_classifier_cifar10_model.MetaClassifierCifar10Model()
                    elif opt.set_arch=="mnist_lenet":
                        classifier = lenet.LeNet5()
                    elif opt.set_arch=="ulp_vgg":
                        classifier = ULP_model.CNN_classifier()
                    elif opt.set_arch == "vit":
                        classifier = ViTWrapper(num_classes=opt.num_classes, in_channels=opt.input_channel,
                                                img_size=opt.input_height, patch_size=4)
                    else:
                        raise ValueError(f"Invalid arch: {opt.set_arch}")
                else:
                    raise ValueError("No architecture specified and could not auto-detect from checkpoint")

                # Try to load the state dict
                try:
                    classifier.load_state_dict(actual_state_dict)
                except RuntimeError as e:
                    print(f"Warning: Could not load state_dict directly. Error: {e}")
                    # Try with strict=False to load matching keys only
                    classifier.load_state_dict(actual_state_dict, strict=False)
        else:
            # No checkpoint path, use specified architecture
            if opt.set_arch:
                if opt.set_arch == "resnet18":
                    classifier = resnet18(num_classes = opt.num_classes, in_channels = opt.input_channel)
                elif opt.set_arch=="preact_resnet18":
                    classifier = preact_resnet.PreActResNet18(num_classes=opt.num_classes)
                elif opt.set_arch=="meta_classifier_cifar10_model":
                    classifier = meta_classifier_cifar10_model.MetaClassifierCifar10Model()
                elif opt.set_arch=="mnist_lenet":
                    classifier = lenet.LeNet5()
                elif opt.set_arch=="ulp_vgg":
                    classifier = ULP_model.CNN_classifier()
                elif opt.set_arch == "vit":
                    classifier = ViTWrapper(num_classes=opt.num_classes, in_channels=opt.input_channel,
                                            img_size=opt.input_height, patch_size=4)
                else:
                    raise ValueError(f"Invalid arch: {opt.set_arch}")
            else:
                raise ValueError("No model path or architecture specified")

        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(opt.device)

class Recorder:
    def __init__(self, opt):
        super().__init__()
        self.mixed_value_best = float("inf")

def test_ori(opt, regression_model, testloader, flip=False):
    regression_model.eval()
    regression_model.AE.eval()
    regression_model.classifier.eval()
    total_pred = 0
    true_pred = 0
    cross_entropy = nn.CrossEntropyLoss()
    for inputs,labels in testloader:
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label

        if flip:
            out, features = regression_model.forward_flip_mask(inputs,opt)
        else:
            out, features = regression_model.forward_ori(inputs,opt)
        predictions = out

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == labels).detach()
        loss_ce = cross_entropy(predictions, target_labels)

    print("BA true_pred:",true_pred)
    print("BA total_pred:",total_pred)
    print(
        "BA test acc:",true_pred * 100.0 / total_pred
        )

def test_ori_attack(opt, regression_model, testloader, flip=False):
    regression_model.eval()
    regression_model.AE.eval()
    regression_model.classifier.eval()
    total_pred = 0
    true_pred = 0
    cross_entropy = nn.CrossEntropyLoss()
    for inputs,labels in testloader:

        inputs = inputs.to(opt.device)

        if opt.asr_test_type == "filter":

            t_mean = opt.t_mean.cuda()
            t_std = opt.t_std.cuda()
            GT_img = inputs
            GT_img = (torch.clamp(GT_img*t_std+t_mean, min=0, max=1).detach().cpu().numpy()*255).astype(np.uint8)
            for j in range(GT_img.shape[0]):
                ori_pil_img = Image.fromarray(GT_img[j].transpose((1,2,0)))
                convered_pil_img = pilgram._1977(ori_pil_img)
                GT_img[j] = np.asarray(convered_pil_img).transpose((2,0,1))
            GT_img = GT_img.astype(np.float32)
            GT_img = GT_img/255
            GT_img = torch.from_numpy(GT_img).cuda()
            GT_img = (GT_img - t_mean)/t_std
            inputs = GT_img
        elif opt.asr_test_type == "wanet":
            inputs = F.grid_sample(inputs, opt.grid_temps.repeat(inputs.shape[0], 1, 1, 1), align_corners=True)


        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label

        if flip:
            out, features = regression_model.forward_flip_mask(inputs,opt)
        else:
            out, features = regression_model.forward_ori(inputs,opt)
        predictions = out

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
        loss_ce = cross_entropy(predictions, target_labels)

    print("ASR true_pred:",true_pred)
    print("ASR total_pred:",total_pred)
    print(
        "ASR test acc:",true_pred * 100.0 / total_pred
        )

def fix_neuron_flip(opt,trainloader,testloader,testloader_asr):

    trained_regression_model = opt.trained_regression_model
    trained_regression_model.eval()
    trained_regression_model.AE.eval()
    trained_regression_model.classifier.eval()

    if opt.asr_test_type == "wanet":
        ckpt_path = opt.hand_set_model_path
        state_dict = torch.load(ckpt_path)
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
        grid_temps = (identity_grid + 0.5 * noise_grid / opt.input_height) * 1
        grid_temps = torch.clamp(grid_temps, -1, 1)

        opt.grid_temps = grid_temps

    test_ori(opt, trained_regression_model,testloader,flip=False)
    test_ori_attack(opt, trained_regression_model,testloader_asr,flip=False)

    neuron_finding_strategy = "hyperplane"

    cross_entropy = nn.CrossEntropyLoss()
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        out, features_reversed, x_before_ae, x_after_ae = trained_regression_model.forward_ae(inputs,opt)
        loss_ce_transformed = cross_entropy(out, labels)

        out, features_ori = trained_regression_model.forward_ori(inputs,opt)
        loss_ce_ori = cross_entropy(out, labels)

        feature_dist = torch.nn.MSELoss(reduction='none').cuda()(features_ori,features_reversed).mean(0)
        print(feature_dist)

        if neuron_finding_strategy == "diff":
            values, indices = feature_dist.reshape(-1).topk(int(0.03*torch.numel(feature_dist)), largest=True, sorted=True)
            flip_mask = torch.zeros(feature_dist.reshape(-1).shape).to(opt.device)
            for index in indices:
                flip_mask[index] = 1
            flip_mask = flip_mask.reshape(feature_dist.shape)

        elif neuron_finding_strategy == "hyperplane":
            flip_mask = trained_regression_model.get_raw_mask(opt)

        opt.flip_mask = flip_mask

        print("loss_ce_transformed:",loss_ce_transformed)
        print("loss_ce_ori:",loss_ce_ori)


    test_ori(opt, trained_regression_model,testloader,flip=True)
    test_ori_attack(opt, trained_regression_model,testloader_asr,flip=True)

def train(opt, init_mask):

    data_now = opt.data_now
    opt.weight_p = 1
    opt.weight_acc = 1
    opt.weight_std = 1
    opt.init_mask = init_mask

    recorder = Recorder(opt)
    regression_model = RegressionModel(opt, init_mask).to(opt.device)

    opt.epoch = 400
    if opt.override_epoch:
        opt.epoch = opt.override_epoch

    optimizerR = torch.optim.Adam(regression_model.AE.parameters(),lr=opt.lr,betas=(0.5,0.9))
    optimizerR_mask = torch.optim.Adam([regression_model.mask_tanh],lr=1e-1,betas=(0.5,0.9))

    regression_model.AE.train()
    recorder = Recorder(opt)
    process = train_step

    warm_up_epoch = 100
    for epoch in range(warm_up_epoch):
        process(regression_model, optimizerR, optimizerR_mask, data_now, recorder, epoch, opt, warm_up=True)

    for epoch in range(opt.epoch):
        process(regression_model, optimizerR, optimizerR_mask, data_now, recorder, epoch, opt)

    opt.trained_regression_model = regression_model

    return recorder, opt

def get_range(opt, init_mask):

    test_dataloader = opt.re_dataloader_total_fixed
    inversion_engine = RegressionModel(opt, init_mask).to(opt.device)

    features_list = []
    features_list_class = [[] for i in range(opt.num_classes)]
    for batch_idx, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.to(opt.device)
        out, features = inversion_engine.forward_ori(inputs,opt)
        print(torch.argmax(out,dim=1))

        features_list.append(features)
        for i in range(inputs.shape[0]):
            features_list_class[labels[i].item()].append(features[i].unsqueeze(0))
    all_features = torch.cat(features_list,dim=0)
    opt.all_features = all_features
    print(all_features.shape)

    del features_list
    del test_dataloader

    weight_map_class = []
    for i in range(opt.num_classes):
        feature_mean_class = torch.cat(features_list_class[i],dim=0).mean(0)
        weight_map_class.append(feature_mean_class)

    opt.weight_map_class = weight_map_class
    del all_features
    del features_list_class

def train_step(regression_model, optimizerR, optimizerR_mask, data_now, recorder, epoch, opt, warm_up=False):
    print("Epoch {} - Label: {} | {} - {}:".format(epoch, opt.target_label, opt.dataset, opt.attack_mode))
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    loss_ce_list = []
    loss_dist_list = []
    loss_list = []
    acc_list = []

    p_loss_list = []
    loss_mask_norm_list = []
    loss_std_list = []

    for inputs in data_now:
        regression_model.AE.train()
        regression_model.mask_tanh.requires_grad = False

        optimizerR.zero_grad()

        inputs = inputs.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
        if warm_up:
            predictions, features, x_before_ae, x_after_ae = regression_model.forward_ae(inputs,opt)
        else:
            predictions, features, x_before_ae, x_after_ae, features_ori = regression_model.forward_ae_mask_p(inputs,opt)

        loss_ce = cross_entropy(predictions, target_labels)

        mse_loss = torch.nn.MSELoss(size_average = True).cuda()(x_after_ae,x_before_ae)

        if warm_up:
            dist_loss = torch.cosine_similarity(opt.weight_map_class[opt.target_label].reshape(-1),features.mean(0).reshape(-1),dim=0)
        else:
            dist_loss = torch.cosine_similarity(opt.weight_map_class[opt.target_label].reshape(-1),features_ori.mean(0).reshape(-1),dim=0)

        acc_list_ = []
        minibatch_accuracy_ = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() / sample_num
        acc_list_.append(minibatch_accuracy_)
        acc_list_ = torch.stack(acc_list_)
        avg_acc_G = torch.mean(acc_list_)

        acc_list.append(minibatch_accuracy_)

        p_loss = mse_loss
        p_loss_bound = opt.p_loss_bound
        loss_std_bound = opt.loss_std_bound

        atk_succ_threshold = opt.ae_atk_succ_t

        if opt.ignore_dist:
            dist_loss = dist_loss*0

        if warm_up:
            if (p_loss>p_loss_bound):
                total_loss = loss_ce + p_loss*100
            else:
                total_loss = loss_ce
        else:
            loss_std = (features_ori*regression_model.get_raw_mask(opt)).std(0).sum()
            loss_std = loss_std/(torch.norm(regression_model.get_raw_mask(opt), 1))

            total_loss = dist_loss*5
            if dist_loss<0:
                total_loss = total_loss - dist_loss*5
            if loss_std>loss_std_bound:
                total_loss = total_loss + loss_std*10*(1+opt.weight_std)
            if (p_loss>p_loss_bound):
                total_loss = total_loss + p_loss*10*(1+opt.weight_p)

            if avg_acc_G.item()<atk_succ_threshold:
                total_loss = total_loss + 1*loss_ce*(1+opt.weight_acc)

        total_loss.backward()
        optimizerR.step()

        mask_norm_bound = int(reduce(lambda x,y:x*y,opt.feature_shape)*opt.mask_size)

        if not warm_up:
            for k in range(1):
                regression_model.AE.eval()
                regression_model.mask_tanh.requires_grad = True

                optimizerR_mask.zero_grad()
                predictions, features, x_before_ae, x_after_ae, features_ori = regression_model.forward_ae_mask_p(inputs,opt)
                loss_mask_ce = cross_entropy(predictions, target_labels)
                loss_mask_norm = torch.norm(regression_model.get_raw_mask(opt), opt.use_norm)
                loss_mask_total = loss_mask_ce
                if loss_mask_norm>mask_norm_bound:
                    loss_mask_total = loss_mask_total + loss_mask_norm

                loss_mask_total.backward()
                optimizerR_mask.step()

        loss_ce_list.append(loss_ce.detach())
        loss_dist_list.append(dist_loss.detach())
        loss_list.append(total_loss.detach())

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

        if not warm_up:
            p_loss_list.append(p_loss)
            loss_mask_norm_list.append(loss_mask_norm)
            loss_std_list.append(loss_std)

    loss_ce_list = torch.stack(loss_ce_list)
    loss_dist_list = torch.stack(loss_dist_list)
    loss_list = torch.stack(loss_list)
    acc_list = torch.stack(acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_dist = torch.mean(loss_dist_list)
    avg_loss = torch.mean(loss_list)
    avg_acc = torch.mean(acc_list)

    if not warm_up:
        p_loss_list = torch.stack(p_loss_list)
        loss_mask_norm_list = torch.stack(loss_mask_norm_list)
        loss_std_list = torch.stack(loss_std_list)

        avg_p_loss = torch.mean(p_loss_list)
        avg_loss_mask_norm = torch.mean(loss_mask_norm_list)
        avg_loss_std = torch.mean(loss_std_list)
        print("avg_ce_loss:",avg_loss_ce)
        print("avg_asr:",avg_acc)
        print("avg_p_loss:",avg_p_loss)
        print("avg_loss_mask_norm:",avg_loss_mask_norm)
        print("avg_loss_std:",avg_loss_std)


        if avg_acc.item()<atk_succ_threshold:
            print("@avg_asr lower than bound")
        if avg_p_loss>1.0*p_loss_bound:
            print("@avg_p_loss larger than bound")
        if avg_loss_mask_norm>1.0*mask_norm_bound:
            print("@avg_loss_mask_norm larger than bound")
        if avg_loss_std>1.0*loss_std_bound:
            print("@avg_loss_std larger than bound")


        mixed_value = avg_loss_dist.detach() - avg_acc + max(avg_p_loss.detach()-p_loss_bound,0)/p_loss_bound + max(avg_loss_mask_norm.detach()-mask_norm_bound,0)/mask_norm_bound + max(avg_loss_std.detach()-loss_std_bound,0)/loss_std_bound
        print("mixed_value:",mixed_value)
        if mixed_value < recorder.mixed_value_best:
            recorder.mixed_value_best = mixed_value
        opt.weight_p = max(avg_p_loss.detach()-p_loss_bound,0)/p_loss_bound
        opt.weight_acc = max(atk_succ_threshold-avg_acc,0)/atk_succ_threshold
        opt.weight_std = max(avg_loss_std.detach()-loss_std_bound,0)/loss_std_bound


    print(
        "  Result: ASR: {:.3f} | Cross Entropy Loss: {:.6f} | Dist Loss: {:.6f} | Mixed_value best: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_dist, recorder.mixed_value_best
        )
    )

    recorder.final_asr = avg_acc
    
    return avg_acc

if __name__ == "__main__":
    pass
