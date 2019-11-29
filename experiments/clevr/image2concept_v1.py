#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_nscl_derender.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/10/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Derendering model for the Neuro-Symbolic Concept Learner.

Unlike the model in NS-VQA, the model receives only ground-truth programs and needs to execute the program
to get the supervision for the VSE modules. This model tests the implementation of the differentiable
(or the so-called quasi-symbolic) reasoning process.

Note that, in order to train this model, one must use the curriculum learning.
"""

from jacinle.utils.container import GView
from nscl.models.image2concept_v1 import make_im2concept_v1_configs, Im2Conceptv1Model
from nscl.models.utils import canonize_monitors, update_from_loss_module

configs = make_im2concept_v1_configs()
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True


class Model(Im2Conceptv1Model):
    def __init__(self, args, vocab):
        super().__init__(args, vocab, configs)
        self.loss_ratio = args.loss_ratio
        self.true_mask = args.true_mask

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        f_scene = self.resnet(feed_dict.image) # [batch_size=32,n_channels=256,h=16,w=24]
        f_sng = self.scene_graph(f_scene,feed_dict.image, feed_dict.objects_mask if self.true_mask else None)

        programs = feed_dict.program_qsseq
        programs, buffers, answers = self.reasoning(f_sng, programs, fd=feed_dict)
        outputs['buffers'] = buffers
        outputs['answer'] = answers

        update_from_loss_module(monitors, outputs, self.scene_graph.get_monitor())
        update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))
        canonize_monitors(monitors)

        if self.training:
            loss = monitors['loss/qa'] + monitors['loss/monet'] * self.loss_ratio
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            outputs['buffers'] = buffers
            return outputs

def make_model(args, vocab):
    return Model(args, vocab)
