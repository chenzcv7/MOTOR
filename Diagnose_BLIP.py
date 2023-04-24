import torch
import argparse
import ruamel_yaml as yaml
import numpy as np
from diagnose_api.trainer import Trainer
from models.blip import blip_feature_extractor
from diagnose_api.optimizers import build_lr_scheduler, build_optimizer
from diagnose_api.diagnose_model import DenseNet121
from torch import nn
from blip_original import create_dataset, create_loader, create_sampler
from diagnose_api.visual_extractor import VisualExtractor



def main(args, config):
    # utils.init_distributed_mode(args)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create data loader
    train_dataset, val_dataset, test_dataset = create_dataset('diagnose_%s' % args.dataset_name, args, config)


    samplers = [None, None, None]
    train_dataloader, val_dataloader, test_dataloader = create_loader([train_dataset, val_dataset, test_dataset],
                                                                      samplers,
                                                                      batch_size=[args.batch_size] * 3,
                                                                      num_workers=[4, 4, 4],
                                                                      is_trains=[True, False, False],
                                                                      collate_fns=[None, None, None])

    if args.visual == 'resnet101':
        model = VisualExtractor(args)
        classifier = DenseNet121(num_ftrs=2048, num_medterm=14)

    elif args.visual == 'densenet':
        model = DenseNet121(num_ftrs=1024, num_medterm=14, mode='densenet')
        classifier = DenseNet121(num_ftrs=1024, num_medterm=14)

    elif args.visual == 'BLIP':
        # build model architecture
        model = blip_feature_extractor(pretrained=args.pretrained, image_size=config['image_size'], vit=config['vit'],
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], args=args)
        classifier = DenseNet121(num_ftrs=768, num_medterm=14)



    # get function handles of loss
    criterion = nn.BCELoss()

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, config, model, classifier)
    lr_scheduler = build_lr_scheduler(args, optimizer)


    # build trainer and start to train
    trainer = Trainer(model, classifier, criterion, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader, None)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/BLIP.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--pretrained', default='')
    parser.add_argument('--output_dir', default='output/generation')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    # Data input settings
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/data/linhaokun/project/dataset/MIMIC-CXR/mimic_cxr/images',
                        help='the path to the directory containing the data.')
    parser.add_argument('-ann_path', type=str,
                        default='/mnt/data/chenzicong_new/Medical_BLIP/data/MIMIC_CXR_knowledge_label.json',
                        help='the path to the directory containing the data.')

    parser.add_argument('--chex_image_dir', type=str,
                        default='/data2/chenzicong/CX14_dataset/images',
                        help='the path to the directory containing the data.')
    parser.add_argument('--chex_path', type=str,
                        default='/data2/chenzicong/CX14/chexpert_diagnosis_new.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['chexpert', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=90, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=50, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/fair', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/diagnosis/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='CIDER', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_BLIP', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_classifier', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    # parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--task', type=str, default='diagnosis',
                        choices=['pretrain', 'retrieval', 'genenration', 'diagnosis', 'vqa'],
                        help='the dataset to be used.')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)


    main(args, config)
