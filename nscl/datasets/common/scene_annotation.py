#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_annotation.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/05/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np

import jaclearn.vision.coco.mask_utils as mask_utils


__all__ = ['annotate_objects']


def _is_object_annotation_available(scene):
    if len(scene['objects']) > 0 and 'mask' in scene['objects'][0]:
        return True
    return False


def _get_object_masks(scene):
    """Backward compatibility: in self-generated clevr scenes, the groundtruth masks are provided;
    while in the clevr test data, we use Mask R-CNN to detect all the masks, and stored in `objects_detection`."""
    if 'objects_detection' not in scene:
        return scene['objects']
    if _is_object_annotation_available(scene):
        return scene['objects']
    return scene['objects_detection']


def annotate_objects(scene):
    if 'objects' not in scene and 'objects_detection' not in scene:
        return dict()

    boxes = [mask_utils.toBbox(i['mask']) for i in _get_object_masks(scene)]
    if len(boxes) == 0:
        return {'objects': np.zeros((0, 4), dtype='float32')}
    boxes = np.array(boxes)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    masks = [mask_utils.decode(i['mask']) for i in _get_object_masks(scene)]
    masks = mask_canonize(masks)

    return {'objects': boxes.astype('float32'), 'objects_mask': masks}

def mask_canonize(masks,slot_num=11):
    _all_ = np.sum(np.array(masks,dtype=masks[0].dtype),axis=0,dtype=masks[0].dtype)
    _all_ = (_all_ == 0).astype(_all_.dtype)
    full_masks = [_all_]+masks
    full_masks = full_masks[:slot_num]
    canonized_masks = np.stack(full_masks)
    if canonized_masks.shape[0] < slot_num:
        pad_masks = np.zeros((slot_num - canonized_masks.shape[0],)+canonized_masks.shape[1:],dtype=canonized_masks.dtype)
        canonized_masks = np.concatenate([canonized_masks,pad_masks],axis=0)
    return canonized_masks        