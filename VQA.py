import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizer
# from models.tokenization_bert import BertTokenizer

#import utils
# from dataset.utils import save_result
# from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
#
# from scheduler import create_scheduler
# from optim import create_optimizer

import argparse
from vqa_api.tools.create_dictionary import Dictionary
import os
from torch.utils.data import DataLoader
import vqa_api.utils

# from vqa_api.dataset_RAD import VQAFeatureDataset
# from vqa_api.multi_level_model import BAN_Model
# from vqa_api.train import train

from vqa_api.dataset_RAD_new import VQAFeatureDataset
from vqa_api.multi_level_model_new import BAN_Model
from vqa_api.train_new import train

from blip_original import create_loader
import torch
from vqa_api.classify_question import classify_model

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    #parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    # GPU config
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=4,
                        help='use gpu device. default:0')

    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models',
                        help='save file directory')

    # Training testing or sampling Hyper-parameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='the number of epoches')
    parser.add_argument('--lr', default=1e-4, type=float, metavar='lr',
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')
    parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # # Train with RAD
    parser.add_argument('--use_data', action='store_true', default=True,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--data_dir', type=str,
                        help='RAD dir')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Attention --------------------------------------------------------------------------------------------------------
    # Choices of attention models
    parser.add_argument('--attention', type=str, default='BAN', choices=['BAN'],
                        help='the model we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--glimpse', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Question ---------------------------------------------------------------------------------------------------------
    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='GRU', choices=['LSTM', 'GRU'],
                        help='the RNN we use')
    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--cat', type=bool, default=True,
                        help='concatenated 600-D word embedding')
    parser.add_argument('--hid_dim', type=int, default=1024,
                        help='dim of joint semantic features')

    # Vision -----------------------------------------------------------------------------------------------------------
    # Input visual feature dimension
    parser.add_argument('--v_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=True,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=True,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')

    # other model hyper-parameters
    parser.add_argument('--other_model', action='store_true', default=False,
                        help='End to end model')

    # details
    parser.add_argument('--details', type=str, default='original ')

    # ALBEF
    parser.add_argument('--BLIP', action='store_true', default=True,
                        help='add BLIP model')
    parser.add_argument('--vis_feature', type=str, default='ae',
                        help='maml or ae')
    parser.add_argument('--ALBEF_tok', action='store_true', default=True,
                        help='use ALBEF tokenizer')

    parser.add_argument('--test_visual', action='store_true', default=False,
                        help='vision')
    parser.add_argument('--test_text', action='store_true', default=False,
                        help='text')
    parser.add_argument('--test_all', action='store_true', default=False,
                        help='all')

    parser.add_argument('--test_A', action='store_true', default=False,
                        help='all')
    parser.add_argument('--test_B', action='store_true', default=False,
                        help='all')
    parser.add_argument('--test_C', action='store_true', default=False,
                        help='all')
    parser.add_argument('--add_typeatt1', action='store_true', default=False,
                        help='all')
    parser.add_argument('--add_typeatt2', action='store_true', default=False,
                        help='all')

    parser.add_argument('--bert', type=str, default='base', choices=['base', 'sci', 'cli'],
                        help='the dataset to be used.')

    parser.add_argument('--task', type=str, default='vqa',
                        choices=['pretrain', 'retrieval', 'generation', 'diagnosis', 'vqa'],
                        help='the dataset to be used.')

    parser.add_argument('--pretrained', default='')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--setting', type=str, help='the setting to be used.')
    parser.add_argument('--have_know', default=False, type=bool)
    parser.add_argument('--GK_know', default=False, type=bool)
    parser.add_argument('--concat', default=False, type=bool)
    parser.add_argument('--GK_out', type=str, default='image', choices=['image', 'tag'],
                        help='the dataset to be used.')


    args = parser.parse_args()

    if args.add_typeatt1 or args.add_typeatt2:
        args.test_C = True

    data = './data/'

    args.data_dir = data
    # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # create word dictionary from train+val dataset
    d = Dictionary.load_from_file(data + 'dictionary.pkl')
    # prepare the dataloader
    train_dataset = VQAFeatureDataset('train', args, d, dataroot=data, tokenizer=None)
    eval_dataset = VQAFeatureDataset('test', args, d, dataroot=data, tokenizer=None)
    print('number of training samples: %d' % len(train_dataset))
    samplers = [None, None]
    train_loader, eval_loader = create_loader([train_dataset, eval_dataset],
                                              samplers,
                                              batch_size=[args.batch_size] * 2,
                                              num_workers=[4, 4],
                                              is_trains=[True, False],
                                              collate_fns=[None, None])

    # create VQA model and question classify model
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    model = BAN_Model(train_dataset, args, config)

    question_classify = classify_model(d.ntoken, args.data_dir+'glove6b_init_300d.npy')

    # load the model
    ckpt = args.data_dir.replace('data/','')+'saved_models/type_classifier.pth'

    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        pre_ckpt = torch.load(args.input)
        model.load_state_dict(pre_ckpt.get('model_state', pre_ckpt))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(pre_ckpt.get('optimizer_state', pre_ckpt))
        epoch = pre_ckpt['epoch'] + 1

    train(args, model, question_classify, train_loader, eval_loader)
