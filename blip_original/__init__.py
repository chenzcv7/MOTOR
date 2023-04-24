import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
from torchvision.transforms.functional import InterpolationMode

from .medical_dataset import generation_train, generation_eval, retrieval_train, retrieval_eval, diagnose_train, diagnose_eval, diagnose_knowledge
from .pretrain_dataset import pretrain_dataset


def create_dataset(dataset, args, config):
    #fowllowed by R2Gen
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    if dataset=='pretrain':
        dataset = pretrain_dataset(args.ann_path, args.image_dir, transform_train, args=args)
        return dataset

    if dataset =='generation_iu_xray':
        train_dataset = generation_train(transform_train, args.image_dir.split('&')[0], args.knowledge_path.split('&')[0], prompt=config['prompt'], dataset='iu_xray', args=args)
        val_dataset = generation_eval(transform_test, args.image_dir.split('&')[0], args.ann_path.split('&')[0], 'val', 'iu_xray', args=args)
        test_dataset = generation_eval(transform_test, args.image_dir.split('&')[0], args.ann_path.split('&')[0], 'test', 'iu_xray', args=args)
        return train_dataset, val_dataset, test_dataset

    elif dataset =='generation_mimic_cxr':
        train_dataset = generation_train(transform_train, args.image_dir.split('&')[1], args.knowledge_path.split('&')[1], prompt=config['prompt'], dataset='mimic_cxr', args=args)
        val_dataset = generation_eval(transform_test, args.image_dir.split('&')[1], args.ann_path.split('&')[1], 'val', 'mimic_cxr', args=args)
        test_dataset = generation_eval(transform_test, args.image_dir.split('&')[1], args.ann_path.split('&')[1], 'test', 'mimic_cxr', args=args)
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_iu_xray':
        train_dataset = retrieval_train(transform_train, args.image_dir.split('&')[0], args.knowledge_path.split('&')[0],
                                        dataset='iu_xray', args=args)
        val_dataset = retrieval_eval(transform_test, args.image_dir.split('&')[0], args.ann_path.split('&')[0], 'val',
                                     args=args)
        test_dataset = retrieval_eval(transform_test, args.image_dir.split('&')[0], args.ann_path.split('&')[0], 'test',
                                      args=args)
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_mimic_cxr':
        train_dataset = retrieval_train(transform_train, args.image_dir.split('&')[1], args.knowledge_path.split('&')[1],
                                        dataset='mimic_cxr', args=args)
        val_dataset = retrieval_eval(transform_test, args.image_dir.split('&')[1], args.ann_path.split('&')[1], 'val', args=args)
        test_dataset = retrieval_eval(transform_test, args.image_dir.split('&')[1], args.ann_path.split('&')[1], 'test', args=args)
        return train_dataset, val_dataset, test_dataset

    elif dataset =='diagnose_mimic_cxr':
        train_dataset = diagnose_train(transform_train, args.image_dir, args.ann_path, dataset='mimic_cxr', args=args)
        val_dataset = diagnose_eval(transform_test, args.image_dir, args.ann_path, 'val', 'mimic_cxr', args=args)
        test_dataset = diagnose_eval(transform_test, args.image_dir, args.ann_path, 'test', 'mimic_cxr', args=args)
        return train_dataset, val_dataset, test_dataset

    elif dataset =='diagnose_chexpert':
        train_dataset = diagnose_train(transform_train, args.chex_image_dir, args.chex_path, dataset='chexpert', args=args)
        val_dataset = diagnose_eval(transform_test, args.chex_image_dir, args.chex_path, 'val', 'chexpert', args=args)
        test_dataset = diagnose_eval(transform_test, args.chex_image_dir, args.chex_path, 'test', 'chexpert', args=args)
        return train_dataset, val_dataset, test_dataset

    elif dataset =='diagnose_mimic_knowledge':
        knowledge_dataset = diagnose_knowledge(args.chex_knowledge, dataset='mimic_cxr', args=args)
        return knowledge_dataset
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

