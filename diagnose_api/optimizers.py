import torch


def build_optimizer(args, config, model, classifier):
    # ve_params = list(map(id, model.visual_extractor.parameters()))
    # ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.parameters(), 'lr': config['init_lr']},
         {'params': classifier.parameters(), 'lr': args.lr_classifier}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
