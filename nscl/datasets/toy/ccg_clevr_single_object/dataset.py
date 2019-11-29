#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/06/2019
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os.path as osp
from copy import deepcopy

import numpy as np
from PIL import Image

import jacinle.io as io
import jacinle.random as random
from jacinle.utils.tqdm import tqdm_gofor

from nscl.datasets.definition import gdef
from nscl.datasets.common.filterable import FilterableDatasetUnwrapped, FilterableDatasetView
from nscl.datasets.common.program_translator import nsclseq_to_nsclqsseq

__all__ = ['CCGCLEVRSingleObjectDataset']


class CCGCLEVRSingleObjectDatasetUnwrapped(FilterableDatasetUnwrapped):
    _fields = ['color', 'material', 'shape', 'size']

    def __init__(self, scenes_json, features_h5=None, image_root=None, image_transform=None,
            question_per_image=20, balance_attribute=True, balance_answer=True):
        super().__init__()

        self.scenes_json = scenes_json
        self.features_h5 = features_h5
        self.image_root = image_root
        self.image_transform = image_transform
        self.question_per_image = question_per_image
        self.balance_attribute = balance_attribute
        self.balance_answer = balance_answer

        self.scenes = io.load_json(self.scenes_json)['anns']
        if self.features_h5 is not None:
            self.features = io.open_h5(features_h5, 'r')['features']
            assert len(self.features) == len(self.scenes)

        # TODO(Jiayuan Mao @ 04/10): no vocab for CCG models.
        self.vocab = None

        self._gen_data()

    def _gen_data(self):
        self.questions = list()

        for i, record in tqdm_gofor(self.scenes, desc='Generating questions'):
            for j in range(self.question_per_image):
                question, answer, program = self._gen_data_filter_exist(record)
                self.questions.append(dict(
                    image_index=i,
                    question_id=j,
                    question=question,
                    answer=answer,
                    program=program
                ))

    def _gen_data_filter_exist(self, record):
        if self.balance_attribute:
            attr = random.choice_list(gdef.all_attributes)
            if self.balance_answer:
                answer = bool(random.randint(0, 2))
                if answer:
                    concept = record['object'][attr]
                else:
                    rest_concepts = gdef.attribute_concepts[attr].copy()
                    rest_concepts.remove(record['object'][attr])
                    concept = random.choice_list(rest_concepts)
            else:
                concept = random.choice_list(gdef.attribute_concepts[attr])
                answer = record['object'][attr] == concept
        else:
            # TODO(Jiayuan Mao @ 04/10): implement.
            raise NotImplementedError('Currently not supporting balance_attribute = False')

        if attr == 'shape':
            question = 'any ' + concept
        else:
            question = 'any ' + concept + ' object'

        program = [
            dict(op='scene', inputs=[]),
            dict(op='filter', concept=[concept], inputs=[0]),
            dict(op='exist', inputs=[1]),
        ]
        return question, 'yes' if answer else 'no', program

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        feed_dict = deepcopy(self.get_metainfo(index))
        image_index = feed_dict['image_index']

        if self.image_root is not None:
            feed_dict['image'] = Image.open(osp.join(self.image_root, feed_dict['image_filename'])).convert('RGB')
            feed_dict['image'] = self.image_transform(feed_dict['image'])
        if self.features_h5 is not None:
            feed_dict['feature'] = self.features[image_index]

        feed_dict['question_raw'] = feed_dict.pop('question')
        feed_dict['question_raw_tokenized'] = feed_dict['question_raw'].split()
        feed_dict['question_type'] = 'exist'
        feed_dict['program_seq'] = gdef.program_to_nsclseq(feed_dict.pop('program'))
        feed_dict['program_qsseq'] = nsclseq_to_nsclqsseq(feed_dict['program_seq'])
        feed_dict['answer'] = gdef.canonize_answer(feed_dict['answer'], feed_dict['question_type'])

        return feed_dict

    def _get_metainfo(self, index):
        question = deepcopy(self.questions[index])
        scene = self.scenes[question['image_index']]
        feed_dict = question
        feed_dict['image_filename'] = scene['image_filename']
        for attr in gdef.all_attributes:
            feed_dict[attr] = scene['object'][attr]

        return feed_dict

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def dump_questions(self, dirname):
        io.dump(osp.join(dirname, 'questions.json'), {'questions': self.questions})


class CCGCLEVRSingleObjectDatasetFilterableView(FilterableDatasetView):
    @staticmethod
    def _gen_filter_in_function(field, name_or_set):
        yield lambda x: x[field] in name_or_set
        if type(name_or_set) is str:
            name_or_set = [name_or_set]
        else:
            name_or_set = list(name_or_set)
        yield '{}[{}]'.format(field, '|'.join(name_or_set))

    def filter_color_in(self, name_or_set):
        return self.filter(*self._gen_filter_in_function('color', name_or_set))

    def filter_material_in(self, name_or_set):
        return self.filter(*self._gen_filter_in_function('material', name_or_set))

    def filter_shape_in(self, name_or_set):
        return self.filter(*self._gen_filter_in_function('shape', name_or_set))

    def filter_size_in(self, name_or_set):
        return self.filter(*self._gen_filter_in_function('size', name_or_set))

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.collate import VarLengthCollateV2
        collate_fn = VarLengthCollateV2({
            'image_index': 'skip',
            'image_filename': 'skip',
            'question_raw': 'skip',
            'question_raw_tokenized': 'skip',
            'program_seq': 'skip',
            'program_qsseq': 'skip',
            'answer': 'skip'
        })

        from jactorch.data.dataloader import JacDataLoader
        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=collate_fn
        )


def CCGCLEVRSingleObjectDataset(*args, **kwargs):
    return CCGCLEVRSingleObjectDatasetFilterableView(
            CCGCLEVRSingleObjectDatasetUnwrapped(*args, **kwargs)
    )