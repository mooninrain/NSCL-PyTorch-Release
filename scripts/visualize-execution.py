#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualize-execution.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/19/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Neuro-Symbolic VQA and Neuro-Symbolic Concept Learner.
"""

import time
import os
import os.path as osp

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from PIL import Image
import torchvision.transforms.functional as TF

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar

from jaclearn.visualize.html_table import HTMLTableVisualizer, HTMLTableColumnDesc
from jaclearn.visualize.box import vis_bboxes

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float, as_cpu, as_detached

from nscl.datasets.factory import get_available_datasets, initialize_dataset, get_dataset_builder

logger = get_logger(__file__)

parser = JacArgumentParser(description='')

parser.add_argument('--expr', default=None, metavar='DIR', help='experiment name')
parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--nr-visualize', default=16, type=int, metavar='N')

# supervision and curriculum learning
parser.add_argument('--loss_ratio', default=0.1, type=float)
parser.add_argument('--loss_type', required=True, choices=['joint', 'separate'])
parser.add_argument('--supervision', required=True, choices=['derender', 'all'])
parser.add_argument('--curriculum', required=True, choices=['off', 'scene', 'program', 'all'])
parser.add_argument('--question-transform', default='nscliclr', choices=['nscliclr'])

# finetuning and snapshot
parser.add_argument('--load', required=True, type='checked_file', metavar='FILE', help='load the weights from a pretrained model (default: none)')

# data related
parser.add_argument('--dataset_name', default='clevr', choices=['clevr','clevr_mini','clevr_noisy','clevr_mini_noisy'], help='dataset')
parser.add_argument('--dataset', default='clevr', choices=get_available_datasets(), help='dataset')
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--data-split', required=False, type=float, default=0.75, metavar='F', help='fraction / numer of training samples')
parser.add_argument('--data-vocab-json', type='checked_file', metavar='FILE')
parser.add_argument('--data-scenes-json', type='checked_file', metavar='FILE')
parser.add_argument('--data-questions-json', type='checked_file', metavar='FILE')

parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=False, metavar='B', help='use tensorboard or not')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')


args = parser.parse_args()

if args.data_vocab_json is None:
    args.data_vocab_json = osp.join(args.data_dir, 'vocab.json')

args.data_image_root = osp.join(args.data_dir, 'images')
if args.data_scenes_json is None:
    args.data_scenes_json = osp.join(args.data_dir, 'scenes.json')
if args.data_questions_json is None:
    args.data_questions_json = osp.join(args.data_dir, 'questions.json')

# filenames
args.series_name = args.dataset
args.desc_name = escape_desc_name(args.desc)
args.show_mask = True if args.desc_name.startswith('image2concept') else False
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

desc = load_source(args.desc)
configs = desc.configs
args.configs.apply(configs)

def main():
    args.dump_dir = ensure_path(osp.join('dumps', args.dataset_name, args.desc_name, args.expr))
    args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
    args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
    args.vis_dir = osp.join(args.dump_dir, 'vis', args.run_name)

    initialize_dataset(args.dataset)
    build_dataset = get_dataset_builder(args.dataset)

    dataset = build_dataset(args, configs, args.data_image_root, args.data_scenes_json, args.data_questions_json)

    dataset_split = int(len(dataset) * args.data_split) if args.data_split <= 1 else int(args.data_split)
    train_dataset, validation_dataset = dataset.split_trainval(dataset_split)

    logger.critical('Building the model.')
    model = desc.make_model(args, train_dataset.unwrapped.vocab)

    if args.use_gpu:
        model.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            # from jactorch.parallel import UserScatteredJacDataParallel as JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    if args.load:
        from jactorch.io import load_weights
        if load_weights(model, args.load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))

    from jacinle.utils.meter import GroupMeters
    meters = GroupMeters()

    if args.embed:
        from IPython import embed; embed()

    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=True, drop_last=False, nr_workers=args.data_workers)

    model.eval()
    validate_epoch(0, model, validation_dataloader, meters)
    logger.critical(meters.format_simple('Validation', {k: v for k, v in meters.avg.items() if v != 0}, compressed=False))
    return meters


def validate_epoch(epoch, model, val_dataloader, meters, meter_prefix='validation'):
    end = time.time()

    visualized = 0
    vis = HTMLTableVisualizer(args.vis_dir, 'NSCL Execution Visualization')
    vis.begin_html()

    try:
        with tqdm_pbar(total=len(val_dataloader)) as pbar:
            for feed_dict in val_dataloader:
                if args.use_gpu:
                    if not args.gpu_parallel:
                        feed_dict = async_copy_to(feed_dict, 0)

                data_time = time.time() - end; end = time.time()

                output_dict = model(feed_dict)
                monitors = {meter_prefix + '/' + k: v for k, v in as_float(output_dict['monitors']).items()}
                step_time = time.time() - end; end = time.time()

                n = feed_dict['image'].size(0)
                meters.update(monitors, n=n)
                meters.update({'time/data': data_time, 'time/step': step_time})

                feed_dict = GView(as_detached(as_cpu(feed_dict)))
                output_dict = GView(as_detached(as_cpu(output_dict)))

                for i in range(n):
                    with vis.table('Visualize #{} Metainfo'.format(visualized), [
                        HTMLTableColumnDesc('id', 'QID', 'text', {'width': '50px'}),
                        HTMLTableColumnDesc('image', 'Image', 'figure', {'width': '300px'}),
                        HTMLTableColumnDesc('mask', 'Mask', 'figure', {'width': '300px'}),
                        HTMLTableColumnDesc('qa', 'QA', 'text', {'width': '200px'}),
                        HTMLTableColumnDesc('p', 'Program', 'code', {'width': '200px'})
                    ]):
                        image_filename = osp.join(args.data_image_root, feed_dict.image_filename[i])
                        image = Image.open(image_filename)

                        fig, ax = vis_bboxes(image, feed_dict.objects_raw[i], 'object', add_text=False)
                        _ = ax.set_title('object bounding box annotations')

                        if not args.show_mask:
                            montage=fig
                        else:
                            num_slots = output_dict['monet/m'].shape[1]
                            monet_fig = [
                            [output_dict['monet/m'][i,k] for k in range(num_slots)],
                            [output_dict['monet/x'][i,k] for k in range(num_slots)],
                            [output_dict['monet/xm'][i,k] for k in range(num_slots)],
                            [output_dict['monet/x_tilde'][i] for k in range(num_slots)]
                            ]
                            montage = image_compose(monet_fig)

                        QA_string = """
                            <p><b>Q</b>: {}</p>
                            <p><b>A</b>: {}</p>
                        """.format(feed_dict.question_raw[i], feed_dict.answer[i])
                        P_string = '\n'.join([repr(x) for x in feed_dict.program_seq[i]])

                        vis.row(id=i, image=fig, mask=montage, qa=QA_string, p=P_string)
                        plt.close()

                    with vis.table('Visualize #{} Trace'.format(visualized), [
                        HTMLTableColumnDesc('id', 'Step', 'text', {'width': '50px'}),
                        HTMLTableColumnDesc('image', 'Image', 'figure', {'width': '600px'}),
                        HTMLTableColumnDesc('p', 'operation', 'text', {'width': '200px'}),
                        HTMLTableColumnDesc('r', 'result', 'code', {'width': '200px'})
                    ]):
                        # TODO(Jiayuan Mao @ 11/20): support output_dict.programs.
                        for j, (prog, buf) in enumerate(zip(feed_dict.program_seq[i], output_dict.buffers[i])):
                            if j != len(feed_dict.program_seq[i]) - 1 and (buf > 0).long().sum().item() > 0:
                                assert buf.size(0) == feed_dict.objects_raw[i].shape[0]
                                this_objects = feed_dict.objects_raw[i][torch.nonzero(buf > 0)[:, 0].numpy()]
                                fig, ax = vis_bboxes(image, this_objects, 'object', add_text=False)
                            else:
                                fig, ax = vis_bboxes(image, [], 'object', add_text=False)
                            vis.row(id=j, image=fig, p=repr(prog), r=repr(buf))
                            plt.close()

                    visualized += 1
                    if visualized > args.nr_visualize:
                        raise StopIteration()

                pbar.set_description(meters.format_simple(
                    'Epoch {} (validation)'.format(epoch),
                    {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 1},
                    compressed=True
                ))
                pbar.update()

                end = time.time()
    except StopIteration:
        pass


    from jacinle.utils.meta import dict_deep_kv
    from jacinle.utils.printing import kvformat
    with vis.table('Info', [HTMLTableColumnDesc('name', 'Name', 'code', {}), HTMLTableColumnDesc('info', 'KV', 'code', {})]):
        vis.row(name='args', info=kvformat(args.__dict__, max_key_len=32))
        vis.row(name='configs', info=kvformat(dict(dict_deep_kv(configs)), max_key_len=32))
    vis.end_html()

    logger.info('Happy Holiday! You can find your result at "http://monday.csail.mit.edu/xiuming' + osp.realpath(args.vis_dir) + '".')

def image_compose(images,size_h=64,size_w=64):
    # images: list of list

    row = len(images)
    column = len(images[0])
    to_image = Image.new('RGB', (column * size_w, row * size_h))
    for y in range(row):
        for x in range(column):
            from_image = images[y][x].resize((size_w,size_h),Image.BILINEAR)
            to_image.paste(from_image, (x * size_w, y * size_h))
    return to_image



if __name__ == '__main__':
    main()