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
from . import module

from nscl.nn.utils import get_memory

__all__ = ['scene_graph_with_monet']



class scene_graph_with_monet_v1(nn.Module):
    def __init__(self, feature_dim, output_dims, loss_type, pretrained_monet=None, _freeze_=False):
        super().__init__()
        self.loss_type = loss_type

        self.h_f, self.w_f = 16, 24
        self.h_m, self.w_m = 64, 64
        self.h_raw ,self.w_raw = 256, 384
        self.h_raw_raw, self.w_raw_raw = 320, 480
        self.slot_num = 11

        self.feature_dim = feature_dim
        self.output_dims = output_dims

        self.image_resize = module.resize_module_cv2(h1=self.h_raw,w1=self.w_raw,h2=self.h_m,w2=self.w_m)
        self.monet_mask_extract = monet.MONet(pretrained_monet,_freeze_)
        self.true_mask_resize = module.resize_module_cv2(h1=self.h_raw_raw,w1=self.w_raw_raw,h2=self.h_m,w2=self.w_m)
        self.mask_resize = module.resize_module(h1=self.h_m,w1=self.w_m,h2=self.h_f,w2=self.w_f)

        self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
        self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)

        self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
        self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)

        self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1]*self.h_f*self.w_f, output_dims[1]))
        self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2]*self.h_f*self.w_f, output_dims[2]))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if 'bias' in m.__dict__:
                    m.bias.data.zero_() 
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if 'bias' in m.__dict__:
                    m.bias.data.zero_() 

    def forward(self, input, image, masks=None):
        object_features = input #[batch_size,feature_dim,h_f,w_f]
        context_features = self.context_feature_extract(input) #[batch_size,feature_dim,h_f,w_f]
        relation_features = self.relation_feature_extract(input) #[batch_size,feature_dim//2*3,h_f,w_f]

        if masks is None:
            masks = self.monet_mask_extract(self.image_resize(image),masks) # [batch_size,slot_num,h_m,w_m]
        else:
            masks = self.true_mask_resize(masks.view(-1,self.slot_num,self.h_raw_raw,self.w_raw_raw))
        if self.loss_type == 'separate':
            masks = masks.detach()
        masks = self.mask_resize(masks.view(-1,1,self.h_m,self.w_m)).view(input.shape[0],-1,self.h_f,self.w_f)

        sub_id, obj_id = jactorch.meshgrid(torch.arange(self.slot_num, dtype=torch.long, device=input.device), dim=0)
        sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)

        masked_object_features = object_features.unsqueeze(1) * masks.unsqueeze(2) #[batch_size,slot_num,feature_dim,h_f,w_f]
        masked_context_features = context_features.unsqueeze(1) * masks.unsqueeze(2)
        masked_relation_features = relation_features.unsqueeze(1) * (masks[:,sub_id]+masks[:,obj_id]).unsqueeze(2)

        x_context,y_context = masked_context_features.chunk(2,dim=2)
        combined_object_features = torch.cat([masked_object_features,x_context,y_context*masks.unsqueeze(2)],dim=2)
        combined_object_features = combined_object_features.view(-1,self.feature_dim*2,self.h_f,self.w_f)
        combined_object_features = self.object_feature_fuse(combined_object_features)
        combined_object_features = combined_object_features.view(input.shape[0],self.slot_num,self.output_dims[1],self.h_f,self.w_f)

        x_relation,y_relation,z_relation = masked_relation_features.chunk(3,dim=2)
        combined_relation_features = torch.cat([combined_object_features[:,sub_id],combined_object_features[:,obj_id],
            x_relation,y_relation*masks[:, sub_id].unsqueeze(2),z_relation*masks[:,obj_id].unsqueeze(2)],dim=2)
        combined_relation_features = combined_relation_features.view(-1,self.feature_dim // 2 * 3 + self.output_dims[1] * 2,self.h_f,self.w_f)
        combined_relation_features = self.relation_feature_fuse(combined_relation_features)

        combined_object_features = combined_object_features.view(masks.shape[0]*masks.shape[1],-1)
        combined_object_features = self._norm(self.object_feature_fc(combined_object_features))
        combined_object_features = combined_object_features.view(masks.shape[0],masks.shape[1],-1)

        combined_relation_features = combined_relation_features.view(masks.shape[0]*masks.shape[1]**2,-1)
        combined_relation_features = self._norm(self.object_feature_fc(combined_relation_features))
        combined_relation_features = combined_relation_features.view(masks.shape[0],masks.shape[1],masks.shape[1],-1)

        outputs = []
        for i in range(input.shape[0]):
            outputs.append([
                None,
                combined_object_features[i],
                combined_relation_features[i]
            ])
        return outputs

    def get_monitor(self):
        return self.monet_mask_extract.get_monitor()

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)



class scene_graph_with_monet_v2(nn.Module):
    def __init__(self, feature_dim, output_dims, loss_type, pretrained_monet=None, _freeze_=False):
        super().__init__()
        self.loss_type = loss_type

        self.h_f, self.w_f = 16, 24
        self.h_m, self.w_m = 64, 64
        self.h_fc, self.w_fc = 7, 7
        self.h_raw ,self.w_raw = 256, 384
        self.h_raw_raw, self.w_raw_raw = 320, 480
        self.slot_num = 11

        self.mask_feature_dim = 32
        self.feature_dim = feature_dim
        self.output_dims = output_dims

        self.image_resize = module.resize_module_cv2(h1=self.h_raw,w1=self.w_raw,h2=self.h_m,w2=self.w_m)
        self.monet_mask_extract = monet.MONet(pretrained_monet,_freeze_)
        self.true_mask_resize = module.resize_module_cv2(h1=self.h_raw_raw,w1=self.w_raw_raw,h2=self.h_m,w2=self.w_m)
        self.mask_resize = module.resize_module(h1=self.h_m,w1=self.w_m,h2=self.h_f,w2=self.w_f)

        self.mask_feature_extract = nn.Conv2d(1, self.mask_feature_dim, 1)
        self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
        self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)

        self.object_feature_fuse = nn.Conv2d(feature_dim * 2 + self.mask_feature_dim, output_dims[1], 1)
        self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2 + self.mask_feature_dim * 2, output_dims[2], 1)

        self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1]*self.h_f*self.w_f, output_dims[1]))
        self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2]*self.h_f*self.w_f, output_dims[2]))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if 'bias' in m.__dict__:
                    m.bias.data.zero_() 
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if 'bias' in m.__dict__:
                    m.bias.data.zero_() 

    def forward(self, input, image, masks=None):
        object_features = input #[batch_size,feature_dim,h_f,w_f]
        context_features = self.context_feature_extract(input) #[batch_size,feature_dim,h_f,w_f]
        relation_features = self.relation_feature_extract(input) #[batch_size,feature_dim//2*3,h_f,w_f]

        if masks is None:
            masks = self.monet_mask_extract(self.image_resize(image)) # [batch_size,slot_num,h_m,w_m]
        else:
            masks = self.true_mask_resize(masks.view(-1,self.slot_num,self.h_raw_raw,self.w_raw_raw))

        if self.loss_type == 'separate':
            masks = masks.detach()
        masks = self.mask_resize(masks.view(input.shape[0]*self.slot_num,1,self.h_m,self.w_m)).view(input.shape[0],self.slot_num,self.h_f,self.w_f)
        mask_features = self.mask_feature_extract(masks.view(input.shape[0]*self.slot_num,1,self.h_f,self.w_f))
        mask_features = mask_features.view(input.shape[0],self.slot_num,self.mask_feature_dim,self.h_f,self.w_f)

        sub_id, obj_id = jactorch.meshgrid(torch.arange(self.slot_num, dtype=torch.long, device=input.device), dim=0)
        sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)

        masked_object_features = object_features.unsqueeze(1) * masks.unsqueeze(2) #[batch_size,slot_num,feature_dim,h_f,w_f]
        masked_context_features = context_features.unsqueeze(1) * masks.unsqueeze(2)
        masked_relation_features = relation_features.unsqueeze(1) * (masks[:,sub_id]+masks[:,obj_id]).unsqueeze(2)

        x_context,y_context = masked_context_features.chunk(2,dim=2)
        combined_object_features = torch.cat([masked_object_features,x_context,y_context*masks.unsqueeze(2),mask_features],dim=2)
        combined_object_features = combined_object_features.view(input.shape[0]*self.slot_num,-1,self.h_f,self.w_f)
        combined_object_features = self.object_feature_fuse(combined_object_features)
        combined_object_features = combined_object_features.view(input.shape[0],self.slot_num,-1,self.h_f,self.w_f)

        x_relation,y_relation,z_relation = masked_relation_features.chunk(3,dim=2)
        combined_relation_features = torch.cat([combined_object_features[:,sub_id],combined_object_features[:,obj_id],
            x_relation,y_relation*masks[:, sub_id].unsqueeze(2),z_relation*masks[:,obj_id].unsqueeze(2),
            mask_features[:, sub_id], mask_features[:, obj_id]],dim=2)
        combined_relation_features = combined_relation_features.view(input.shape[0]*self.slot_num**2,-1,self.h_f,self.w_f)
        combined_relation_features = self.relation_feature_fuse(combined_relation_features)

        combined_object_features = combined_object_features.view(masks.shape[0]*masks.shape[1],-1)
        combined_object_features = self._norm(self.object_feature_fc(combined_object_features))
        combined_object_features = combined_object_features.view(masks.shape[0],masks.shape[1],-1)

        combined_relation_features = combined_relation_features.view(masks.shape[0]*masks.shape[1]**2,-1)
        combined_relation_features = self._norm(self.object_feature_fc(combined_relation_features))
        combined_relation_features = combined_relation_features.view(masks.shape[0],masks.shape[1],masks.shape[1],-1)

        outputs = []
        for i in range(input.shape[0]):
            outputs.append([
                None,
                combined_object_features[i],
                combined_relation_features[i]
            ])
        return outputs

    def get_monitor(self):
        return self.monet_mask_extract.get_monitor()

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)