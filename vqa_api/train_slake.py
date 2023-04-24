# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         train
# Description:
# Author:       Boliu.Kelvin
# Date:         2020/4/8
# -------------------------------------------------------------------------------
import os
import time
import torch
import vqa_api.utils
from datetime import datetime
import torch.nn as nn
from torch.optim import lr_scheduler
from models.blip import init_tokenizer


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp


# Train phase
def train(args, model, question_model, train_loader, eval_loader, s_opt=None, s_epoch=0):
    device = args.device
    model = model.to(device)
    question_model = question_model.to(device)
    # create packet for output
    vqa_api.utils.create_dir(args.output)
    ckpt_path = args.output
    vqa_api.utils.create_dir(ckpt_path)
    # create logger
    logger = vqa_api.utils.Logger(os.path.join(ckpt_path, args.setting+'_'+'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())
    optim = torch.optim.Adamax(params=model.parameters(), lr=args.lr)

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    best_eval_score = 0
    best_epoch = 0

    tokenizer = init_tokenizer()
    if args.resume is not None:
        resume_path = str(args.resume)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        s_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optim.load_state_dict(checkpoint['optimizer_state'])

        print("Checkpoint loaded. Resume training from epoch {}".format(s_epoch))
    # Epoch passing in training phase
    for epoch in range(s_epoch, args.epochs):
        total_loss = 0
        train_score = 0
        number = 0
        model.train()

        # Predicting and computing score
        for i, (v, q, a, answer_type, question_type, phrase_type, answer_target) in enumerate(train_loader):
            # lr_decay.step()

            if args.maml:
                v[0] = v[0].to(device)
            if args.autoencoder:
                v[1] = v[1].to(device)
            if args.other_model:
                v = v.to(device)

            a = a.to(device)

            q1 = tokenizer(q, padding='longest', truncation=True, max_length=35, return_tensors="pt").to(device)
            q_mask = q1.attention_mask
            q = q1.input_ids

            last_output_close, last_output_open, a_close, a_open = model(v, q, a, answer_target, q_mask)  # last_close 30 * 768 last_open 34 * 768 open


            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            loss_close, loss_open=0,0
            # loss
            if 0 in answer_target:
                loss_close = criterion(preds_close.float(), a_close)
            if 1 in answer_target:
                loss_open = criterion(preds_open.float(), a_open)
            loss = loss_close + loss_open

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            # compute the acc for open and close
            if 0 in answer_target:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            if 1 in answer_target:
                batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()
            total_loss += loss.item()
            train_score += batch_close_score + batch_open_score
            number += q.shape[0]


        total_loss /= len(train_loader)
        train_score = 100 * train_score / number
        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, train_score))
        # Evaluation
        if eval_loader is not None:
            eval_score = evaluate_classifier(model, question_model, eval_loader, args, logger)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                model_path = os.path.join(ckpt_path, args.setting+'_'+'best.pth')
                vqa_api.utils.save_model(model_path, model, best_epoch, optim)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))

# Evaluation
def evaluate_classifier(model, pretrained_model, dataloader, args, logger):
    device = args.device
    score = 0
    total = 0
    open_ended = 0.  # 'OPEN'
    score_open = 0.

    closed_ended = 0.  # 'CLOSED'
    score_close = 0.
    model.eval()
    pretrained_model.eval()

    tokenizer = init_tokenizer()

    with torch.no_grad():
        for i, (v, q, a, answer_type, question_type, phrase_type, answer_target) in enumerate(dataloader):
            if args.maml:
                v[0] = v[0].to(device)
            if args.autoencoder:
                v[1] = v[1].to(device)
            if args.other_model:
                v = v.to(device)
            a = a.to(device)

            q1 = tokenizer(q, padding='longest', truncation=True, max_length=35, return_tensors="pt").to(device)
            q_mask = q1.attention_mask
            q = q1.input_ids

            last_output_close, last_output_open, a_close, a_open = model(v, q, a, answer_target, q_mask)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)

            batch_close_score = 0.
            batch_open_score = 0.
            if preds_close.shape[0] != 0:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            if preds_open.shape[0] != 0:
                batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()

            score += batch_close_score + batch_open_score

            size = q.shape[0]
            total += size  # batch number

            open_ended += preds_open.shape[0]
            score_open += batch_open_score

            closed_ended += preds_close.shape[0]
            score_close += batch_close_score

    score = 100 * score / total
    if open_ended > 0:
        open_score = 100 * score_open / open_ended
    if closed_ended > 0:
        close_score = 100 * score_close / closed_ended
    print(total, open_ended, closed_ended)
    logger.info(
        '[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}%'.format(score, open_score, close_score))
    return score

