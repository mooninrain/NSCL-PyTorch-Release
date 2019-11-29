#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : definition.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/17/2019
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
A toy dataset for testing CCG.
"""

import six

from pyccg.chart import WeightedCCGChartParser, printCCGDerivation
from pyccg.lexicon import Lexicon
from pyccg.logic import TypeSystem, Ontology, Expression

from nscl.datasets.definition import DatasetDefinitionBase

__all__ = [
    'CCGCLEVRSingleObjectDefinition',
    'build_ccg_clevr_single_object_dataset',
    'build_symbolic_ccg_clevr_single_object_dataset'
]


class CCGCLEVRSingleObjectDefinition(DatasetDefinitionBase):
    parameter_types = ['concept', 'relational_concept', 'attribute']
    variable_types = ['object', 'object_set']
    return_types = ['concept', 'integer', 'bool']

    operation_signatures = [
        ('scene', [], [], 'object_set'),
        ('filter', ['concept'], ['object_set'], 'object_set'),
        # ('relate', ['relational_concept'], ['object'], 'object_set'),
        # ('relate_attribute_equal', ['attribute'], ['object'], 'object_set'),
        # ('intersect', [], ['object_set', 'object_set'], 'object_set'),
        # ('union', [], ['object_set', 'object_set'], 'object_set'),

        # ('query', ['attribute'], ['object'], 'concept'),
        # ('query_attribute_equal', ['attribute'], ['object', 'object'], 'bool'),
        ('exist', [], ['object_set'], 'bool'),
        # ('count', [], ['object_set'], 'integer'),
        # ('count_less', [], ['object_set', 'object_set'], 'bool'),
        # ('count_equal', [], ['object_set', 'object_set'], 'bool'),
        # ('count_greater', [], ['object_set', 'object_set'], 'bool'),
    ]

    attribute_concepts = {
        'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
        'material': ['rubber', 'metal'],
        'shape': ['cube', 'sphere', 'cylinder'],
        'size': ['small', 'large']
    }

    relational_concepts = {
        # 'spatial_relation': ['left', 'right', 'front', 'behind']
    }

    def annotate_scene(self, scene):
        return dict()

    def annotate_question_metainfo(self, metainfo):
        return dict()

    def annotate_question(self, metainfo):
        return dict()

    def program_to_nsclseq(self, program, question=None):
        return program

    def canonize_answer(self, answer, question_type):
        if answer in ('yes', 'no'):
            answer = (answer == 'yes')
        elif isinstance(answer, six.string_types) and answer.isdigit():
            answer = int(answer)
            assert 0 <= answer <= 10
        return answer

    def update_collate_guide(self, collate_guide):
        pass

    def make_ontology(self, anonymous_constants=False):
        """Construct a pyccg-compatible ontology system based on the type definitions and signatures."""

        types = TypeSystem(list(set(self.parameter_types).union(set(self.variable_types)).union(set(self.return_types))))

        functions = list()
        constants = list()
        for fname, parameter_inputs, variable_inputs, return_type in self.operation_signatures:
            if len(parameter_inputs) == 0 and len(variable_inputs) == 0:
                function = types.new_function(self.escape_funcname(fname), (return_type, ), None)
                functions.append(function)
                # constants.append(types.new_constant(fname, return_type))
            else:
                all_inputs = tuple(variable_inputs + parameter_inputs)
                function = types.new_function(self.escape_funcname(fname), all_inputs + (return_type, ), None)
                functions.append(function)

        for concept_category, concept_list in zip(
            ['concept', 'attribute', 'relational_concept'],
            [self.all_attribute_concepts, self.all_attributes, self.all_relational_concepts]
        ):
            for i, v in enumerate(concept_list):
                constants.append(types.new_constant(
                    v if not anonymous_constants else '{}_{:06d}'.format(concept_category, i + 1),
                    concept_category
                ))
            for i in range(5):
                constants.append(types.new_constant(
                    '{}_{:06d}'.format(concept_category, len(concept_list) + i),
                    concept_category
                ))

        ontology = Ontology(types, functions, constants)

        return ontology

    def make_initial_lexicons(self, ontology, groundtruth=False):
        initial_lex = r"""
        :- S, N
        """

        if groundtruth:
            initial_lex += r"""
            any => S/N {\x.exist_(x)}
            object => N {scene}
            """

            for concept in self.all_attribute_concepts:
                if self.concept2attribute[concept] == 'shape':
                    initial_lex += r"%% => N {(\x.filter(x, %%))(scene)}".replace('%%', concept) + '\n'
                else:
                    initial_lex += r"%% => N/N {\x.filter(x, %%)}".replace('%%', concept) + '\n'
        else:
            initial_lex += r"""
            _dummy_verb => S/N {\x.exist_(x)}
            _dummy_adj => N/N {\x.filter(x, concept_000001)}
            _dummy_noun => N {scene}
            """

        initial_lex = Lexicon.fromstring(initial_lex, ontology, include_semantics=True)

        return initial_lex

    def escape_funcname(self, name):
        if name == 'exist':
            return 'exist_'
        return name

    def unescape_funcname(self, name):
        if name == 'exist_':
            return 'exist'

        return name


def build_ccg_clevr_single_object_dataset(args, configs, image_root, scenes_json, questions_json):
    import torchvision.transforms as T
    image_transform = T.Compose([
        T.Resize(configs.data.image_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO(Jiayuan Mao @ 04/10): support configs.data.features_h5.
    from .dataset import CCGCLEVRSingleObjectDataset
    dataset = CCGCLEVRSingleObjectDataset(
        scenes_json,
        image_root=image_root, image_transform=image_transform
    )

    return dataset


def build_symbolic_ccg_clevr_single_object_dataset(args):
    from .dataset import CCGCLEVRSingleObjectDataset
    dataset = CCGCLEVRSingleObjectDataset(
        args.data_scenes_json,
        image_root=None, image_transform=None
    )

    return dataset