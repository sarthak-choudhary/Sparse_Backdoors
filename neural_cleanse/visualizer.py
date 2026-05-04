#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils_backdoor

from decimal import Decimal


class Visualizer:

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1
    # pixel intensity range of image and preprocessing method
    # raw: [0, 255]
    # mnist: [0, 1]
    # imagenet: imagenet mean centering
    # inception: [-1, 1]
    INTENSITY_RANGE = 'raw'
    # type of regularization of the mask
    REGULARIZATION = 'l1'
    # threshold of attack success rate for dynamically changing cost
    ATTACK_SUCC_THRESHOLD = 0.99
    # patience
    PATIENCE = 10
    # multiple of changing cost, down multiple is the square root of this
    COST_MULTIPLIER = 1.5
    # if resetting cost to 0 at the beginning
    # default is true for full optimization, set to false for early detection
    RESET_COST_TO_ZERO = True
    # min/max of mask
    MASK_MIN = 0
    MASK_MAX = 1
    # min/max of raw pixel intensity
    COLOR_MIN = 0
    COLOR_MAX = 255
    # number of color channel
    IMG_COLOR = 3
    # whether to shuffle during each epoch
    SHUFFLE = True
    # batch size of optimization
    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # whether to return log or not
    RETURN_LOGS = True
    # whether to save last pattern or best pattern
    SAVE_LAST = False
    # epsilon used in tanh
    EPSILON = 1e-7
    # early stop flag
    EARLY_STOP = True
    # early stop threshold
    EARLY_STOP_THRESHOLD = 0.99
    # early stop patience
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    # save tmp masks, for debugging purpose
    SAVE_TMP = False
    # dir to save intermediate masks
    TMP_DIR = 'tmp'
    # whether input image has been preprocessed or not
    RAW_INPUT_FLAG = False

    def __init__(self, model, intensity_range, regularization, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=UPSAMPLE_SIZE,
                 attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG,
                 device='cuda'):

        assert intensity_range in {'imagenet', 'inception', 'mnist', 'raw'}
        assert regularization in {None, 'l1', 'l2'}

        self.model = model
        self.device = device
        self.model.eval()  # Set to eval mode for inference
        self.intensity_range = intensity_range
        self.regularization = regularization
        self.input_shape = input_shape
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag

        # Handle input_shape as (H, W, C) or (H, W)
        if len(input_shape) == 3:
            h, w = input_shape[0], input_shape[1]
        elif len(input_shape) == 2:
            h, w = input_shape[0], input_shape[1]
        else:
            raise ValueError(f"Invalid input_shape: {input_shape}")
        
        mask_size = np.ceil(np.array([h, w], dtype=float) / upsample_size)
        mask_size = mask_size.astype(int)
        self.mask_size = tuple(mask_size)

        # Initialize pattern and mask as parameters
        mask_tanh_init = np.zeros(self.mask_size)
        # Pattern shape: if input_shape is (H, W, C), use it; otherwise assume (H, W, 3)
        if len(input_shape) == 3:
            pattern_tanh_init = np.zeros(input_shape)
        else:
            pattern_tanh_init = np.zeros((h, w, 3))

        # Convert to PyTorch tensors and register as parameters
        self.pattern_tanh = nn.Parameter(
            torch.tensor(pattern_tanh_init, dtype=torch.float32, device=device)
        )
        self.mask_tanh = nn.Parameter(
            torch.tensor(mask_tanh_init, dtype=torch.float32, device=device)
        )

        # Cost parameter
        self.cost = init_cost
        self.cost_tensor = torch.tensor(self.cost, dtype=torch.float32, device=device)

    def _preprocess(self, x_input, intensity_range):
        """Preprocess input images."""
        if intensity_range == 'raw':
            return x_input
        elif intensity_range == 'imagenet':
            # 'RGB'->'BGR'
            x_tmp = x_input[:, [2, 1, 0], :, :]
            # Zero-center by mean pixel
            mean = torch.tensor([[[103.939]], [[116.779]], [[123.68]]], 
                               device=x_input.device, dtype=x_input.dtype)
            x_preprocess = x_tmp - mean
            return x_preprocess
        elif intensity_range == 'inception':
            x_preprocess = (x_input / 255.0 - 0.5) * 2.0
            return x_preprocess
        elif intensity_range == 'mnist':
            x_preprocess = x_input / 255.0
            return x_preprocess
        else:
            raise Exception('unknown intensity_range %s' % intensity_range)

    def _reverse_preprocess(self, x_input, intensity_range):
        """Reverse preprocessing."""
        if intensity_range == 'raw':
            return x_input
        elif intensity_range == 'imagenet':
            mean = torch.tensor([[[103.939]], [[116.779]], [[123.68]]], 
                               device=x_input.device, dtype=x_input.dtype)
            x_reverse = x_input + mean
            # 'BGR'->'RGB'
            x_reverse = x_reverse[:, [2, 1, 0], :, :]
            return x_reverse
        elif intensity_range == 'inception':
            x_reverse = (x_input / 2 + 0.5) * 255.0
            return x_reverse
        elif intensity_range == 'mnist':
            x_reverse = x_input * 255.0
            return x_reverse
        else:
            raise Exception('unknown intensity_range %s' % intensity_range)

    def _get_mask_upsample(self):
        """Get upsampled mask."""
        # Convert mask from tanh space to [0, 1]
        mask = (torch.tanh(self.mask_tanh) / (2 - self.epsilon) + 0.5)
        mask = torch.clamp(mask, self.mask_min, self.mask_max)
        
        # Expand to include color channels
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask_expanded = mask_expanded.repeat(1, self.img_color, 1, 1)  # (1, C, H, W)
        
        # Get target size from input_shape
        if len(self.input_shape) == 3:
            target_h, target_w = self.input_shape[0], self.input_shape[1]
        else:
            target_h, target_w = self.input_shape[0], self.input_shape[1]
        
        # Upsample
        if self.upsample_size > 1:
            mask_upsample = F.interpolate(
                mask_expanded,
                size=(target_h, target_w),
                mode='nearest'
            )
        else:
            # Crop if needed
            h, w = mask_expanded.shape[2], mask_expanded.shape[3]
            if h > target_h or w > target_w:
                mask_upsample = mask_expanded[:, :, :target_h, :target_w]
            else:
                mask_upsample = mask_expanded
        
        return mask_upsample

    def _get_pattern_raw(self):
        """Get pattern in raw pixel space."""
        pattern = (torch.tanh(self.pattern_tanh) / (2 - self.epsilon) + 0.5) * 255.0
        pattern = torch.clamp(pattern, self.color_min, self.color_max)
        # Ensure shape is (1, C, H, W) in NCHW format
        if len(pattern.shape) == 3:
            # pattern is (H, W, C) in NHWC format, convert to (1, C, H, W)
            pattern = pattern.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        elif len(pattern.shape) == 4:
            # Check if it's (1, H, W, C) and convert to (1, C, H, W)
            if pattern.shape[-1] in [1, 3] and pattern.shape[1] not in [1, 3]:
                pattern = pattern.permute(0, 3, 1, 2)  # (1, H, W, C) -> (1, C, H, W)
        return pattern

    def reset_opt(self, optimizer):
        """Reset optimizer state."""
        optimizer.state.clear()

    def reset_state(self, pattern_init, mask_init):
        """Reset pattern and mask to initial values."""
        print('resetting state')

        # Setting cost
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost
        self.cost_tensor = torch.tensor(self.cost, dtype=torch.float32, device=self.device)

        # Setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, self.mask_min, self.mask_max)
        pattern = np.clip(pattern, self.color_min, self.color_max)

        # Convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        with torch.no_grad():
            self.mask_tanh.data = torch.tensor(mask_tanh, dtype=torch.float32, device=self.device)
            self.pattern_tanh.data = torch.tensor(pattern_tanh, dtype=torch.float32, device=self.device)

    def save_tmp_func(self, step):
        """Save temporary mask and fusion for debugging."""
        mask_upsample = self._get_mask_upsample()
        pattern_raw = self._get_pattern_raw()
        
        cur_mask = mask_upsample[0, 0, :, :].cpu().detach().numpy()
        img_filename = '%s/%s' % (self.tmp_dir, 'tmp_mask_step_%d.png' % step)
        utils_backdoor.dump_image(np.expand_dims(cur_mask, axis=2) * 255,
                                  img_filename, 'png')

        cur_fusion = (mask_upsample * pattern_raw)[0].cpu().detach().numpy()
        cur_fusion = np.transpose(cur_fusion, (1, 2, 0))  # CHW -> HWC
        img_filename = '%s/%s' % (self.tmp_dir, 'tmp_fusion_step_%d.png' % step)
        utils_backdoor.dump_image(cur_fusion, img_filename, 'png')

    def visualize(self, gen, y_target, pattern_init, mask_init):
        """Main visualization function."""
        # Reset state
        self.reset_state(pattern_init, mask_init)

        # Setup optimizer
        optimizer = optim.Adam([self.pattern_tanh, self.mask_tanh], lr=self.lr, betas=(0.5, 0.9))

        # Best optimization results
        mask_best = None
        mask_upsample_best = None
        pattern_best = None
        reg_best = float('inf')

        # Logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # Counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        # Convert generator to DataLoader if needed
        if isinstance(gen, DataLoader):
            data_loader = gen
            data_iter = iter(data_loader)
        else:
            # Legacy generator (numpy arrays)
            data_iter = gen

        # Loop start
        for step in range(self.steps):
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []

            for idx in range(self.mini_batch):
                # Get batch
                try:
                    X_batch, _ = next(data_iter)
                    
                    # Convert to tensor if needed
                    if isinstance(X_batch, np.ndarray):
                        X_batch = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
                    else:
                        X_batch = X_batch.to(self.device)
                    
                    # Ensure correct format: (N, C, H, W)
                    if len(X_batch.shape) == 4:
                        if X_batch.shape[-1] in [1, 3]:  # NHWC format
                            X_batch = X_batch.permute(0, 3, 1, 2)  # Convert to NCHW
                except StopIteration:
                    # Reset iterator
                    if isinstance(gen, DataLoader):
                        data_iter = iter(data_loader)
                    else:
                        data_iter = iter(gen)
                    X_batch, _ = next(data_iter)
                    
                    if isinstance(X_batch, np.ndarray):
                        X_batch = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
                    else:
                        X_batch = X_batch.to(self.device)
                    
                    # Ensure correct format: (N, C, H, W)
                    if len(X_batch.shape) == 4:
                        if X_batch.shape[-1] in [1, 3]:  # NHWC format
                            X_batch = X_batch.permute(0, 3, 1, 2)  # Convert to NCHW

                batch_size = X_batch.shape[0]
                Y_target = torch.full((batch_size,), y_target, dtype=torch.long, device=self.device)

                # Get mask and pattern
                mask_upsample = self._get_mask_upsample()
                pattern_raw = self._get_pattern_raw()

                # Prepare input
                # X_batch is in NCHW format, convert to NHWC for preprocessing if needed
                if self.intensity_range == 'raw' and not self.raw_input_flag:
                    # For raw, we can work directly with NCHW
                    input_raw = X_batch
                else:
                    # Convert NCHW to NHWC for preprocessing
                    X_batch_nhwc = X_batch.permute(0, 2, 3, 1)  # NCHW -> NHWC
                    if self.raw_input_flag:
                        input_raw = X_batch_nhwc
                    else:
                        input_raw = self._reverse_preprocess(X_batch_nhwc, self.intensity_range)

                # Apply mask operation in raw domain
                # mask_upsample and pattern_raw are in (1, C, H, W) format
                # input_raw might be in NHWC or NCHW depending on intensity_range
                # Determine format: NHWC has channels in last dim, NCHW has channels in dim 1
                is_nhwc = (len(input_raw.shape) == 4 and 
                          input_raw.shape[-1] in [1, 3] and 
                          input_raw.shape[1] not in [1, 3])
                
                if is_nhwc:  # NHWC format: (N, H, W, C)
                    # Convert mask and pattern to NHWC
                    mask_nhwc = mask_upsample.permute(0, 2, 3, 1)  # (1, C, H, W) -> (1, H, W, C)
                    pattern_nhwc = pattern_raw.permute(0, 2, 3, 1)  # (1, C, H, W) -> (1, H, W, C)
                    reverse_mask_nhwc = 1 - mask_nhwc
                    X_adv_raw = reverse_mask_nhwc * input_raw + mask_nhwc * pattern_nhwc
                    # Convert back to NCHW for model
                    X_adv_raw = X_adv_raw.permute(0, 3, 1, 2)  # NHWC -> NCHW
                else:  # NCHW format: (N, C, H, W)
                    # Ensure mask_upsample and pattern_raw match batch size for broadcasting
                    reverse_mask = 1 - mask_upsample
                    # Broadcast (1, C, H, W) to match (batch_size, C, H, W)
                    X_adv_raw = reverse_mask * input_raw + mask_upsample * pattern_raw

                # Preprocess
                if self.intensity_range == 'raw':
                    X_adv = X_adv_raw
                else:
                    # Convert to NHWC for preprocessing, then back to NCHW
                    X_adv_raw_nhwc = X_adv_raw.permute(0, 2, 3, 1)  # NCHW -> NHWC
                    X_adv_nhwc = self._preprocess(X_adv_raw_nhwc, self.intensity_range)
                    X_adv = X_adv_nhwc.permute(0, 3, 1, 2)  # NHWC -> NCHW

                # Forward pass
                optimizer.zero_grad()
                output = self.model(X_adv)

                # Losses
                loss_ce = F.cross_entropy(output, Y_target)
                
                if self.regularization is None:
                    loss_reg = torch.tensor(0.0, device=self.device)
                elif self.regularization == 'l1':
                    loss_reg = torch.sum(torch.abs(mask_upsample)) / self.img_color
                elif self.regularization == 'l2':
                    loss_reg = torch.sqrt(torch.sum(mask_upsample ** 2) / self.img_color)
                else:
                    loss_reg = torch.tensor(0.0, device=self.device)

                loss = loss_ce + loss_reg * self.cost_tensor

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accuracy
                pred = output.argmax(dim=1)
                acc = (pred == Y_target).float().mean()

                loss_ce_list.append(loss_ce.item())
                loss_reg_list.append(loss_reg.item())
                loss_list.append(loss.item())
                loss_acc_list.append(acc.item())

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)

            # Check to save best mask or not
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_upsample = self._get_mask_upsample()
                pattern_raw = self._get_pattern_raw()
                
                mask_best = mask_upsample[0, 0, :, :].cpu().detach().numpy()
                mask_upsample_best = mask_upsample[0, 0, :, :].cpu().detach().numpy()
                pattern_best = pattern_raw[0].cpu().detach().numpy()
                pattern_best = np.transpose(pattern_best, (1, 2, 0))  # CHW -> HWC
                reg_best = avg_loss_reg

            # Verbose
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(self.cost), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            # Save log
            logs.append((step,
                         avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc,
                         reg_best, self.cost))

            # Check early stop
            if self.early_stop:
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # Check cost modification
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    self.cost_tensor = torch.tensor(self.cost, dtype=torch.float32, device=self.device)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                self.cost_tensor = torch.tensor(self.cost, dtype=torch.float32, device=self.device)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                self.cost_tensor = torch.tensor(self.cost, dtype=torch.float32, device=self.device)
                cost_down_flag = True

            if self.save_tmp:
                self.save_tmp_func(step)

        # Save the final version
        if mask_best is None or self.save_last:
            mask_upsample = self._get_mask_upsample()
            pattern_raw = self._get_pattern_raw()
            
            mask_best = mask_upsample[0, 0, :, :].cpu().detach().numpy()
            mask_upsample_best = mask_upsample[0, 0, :, :].cpu().detach().numpy()
            pattern_best = pattern_raw[0].cpu().detach().numpy()
            pattern_best = np.transpose(pattern_best, (1, 2, 0))  # CHW -> HWC

        if self.return_logs:
            return pattern_best, mask_best, mask_upsample_best, logs
        else:
            return pattern_best, mask_best, mask_upsample_best
