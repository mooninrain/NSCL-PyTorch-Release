#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/19/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Scene Graph generation.
"""

import os

import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn

from . import monet

__all__ = ['SceneGraph_MONet']


class SceneGraph_MONet(nn.Module):
    def __init__(self, feature_dim, output_dims, mask_downsample_rate):
        super().__init__()
        self.slot_num = 11
        self.pool_size = 7

        self.feature_dim = feature_dim
        self.output_dims = output_dims

        self.monet_mask_extract = monet.MONet()
        self.mask_downsample_rate = mask_downsample_rate

        self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
        self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)

        self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
        self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)

        self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1] * self.pool_size ** 2, output_dims[1]))
        self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2] * self.pool_size ** 2, output_dims[2]))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input):
        object_features = input
        context_features = self.context_feature_extract(input)
        relation_features = self.relation_feature_extract(input)
        masks = self.monet_mask_extract(input)
        masks = downsample_2D(masks,downsample_rate=self.mask_downsample_rate) #[batch_size,slot_num,h//downsample_rate,w//downsample_rate]

        outputs = list()
        objects_index = 0
        for i in range(input.size(0)):

            this_context_features = context_features * masks[i]


            this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
            x, y = this_context_features.chunk(2, dim=1)
            this_object_features = self.object_feature_fuse(torch.cat([
                self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1)),
                x, y * box_context_imap
            ], dim=1))

            this_relation_features = self.relation_roi_pool(relation_features, torch.cat([rel_batch_ind, union_box], dim=-1))
            x, y, z = this_relation_features.chunk(3, dim=1)
            this_relation_features = self.relation_feature_fuse(torch.cat([
                this_object_features[sub_id], this_object_features[obj_id],
                x, y * sub_union_imap, z * obj_union_imap
            ], dim=1))

            import pdb; pdb.set_trace()
            outputs.append([
                None,
                self._norm(self.object_feature_fc(this_object_features.view(box.size(0), -1))),
                self._norm(self.relation_feature_fc(this_relation_features.view(box.size(0) * box.size(0), -1)).view(box.size(0), box.size(0), -1))
            ])

        return outputs

    def get_loss(self):
        return self.monet_mask_extract.get_loss()

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)

def downsample_2D(input,downsample_rate=4,mode='mean'):
    if len(input.shape!=4):
        raise ValueError('input should be [batch_size,slot_num,h,w]')
    elif input.shape[2]%downsample_rate!=0 or input.shape[3]%downsample_rate!=0:
        raise ValueError('h,w should be divided exactly by downsample_rate')

    input = input.view(input.shape[0],input.shape[1],input.shape[2]//downsample_rate,downsample_rate,input.shape[3]//downsample_rate,downsample_rate)
    if mode == 'mean':
        return torch.mean(input.float(),dim=(3,5))
    elif mode == 'sum':
        return torch.sum(input.float(),dim=(3,5))
    else:
        raise ValueError('improper mode')
    