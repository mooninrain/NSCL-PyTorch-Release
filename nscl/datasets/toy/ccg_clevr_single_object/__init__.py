#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/10/2019
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from nscl.datasets.factory import register_dataset
from .definition import CCGCLEVRSingleObjectDefinition
from .definition import build_ccg_clevr_single_object_dataset, build_symbolic_ccg_clevr_single_object_dataset


register_dataset(
    'toy.ccg_clevr_single_object', CCGCLEVRSingleObjectDefinition,
    builder=build_ccg_clevr_single_object_dataset,
    symbolic_builder=build_symbolic_ccg_clevr_single_object_dataset,
)