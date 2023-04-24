import os
from abc import abstractmethod
import json
import time
import torch
import pandas as pd
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import metrics
from .calculate_metrics import calculate_metrics
from itertools import cycle

class BaseTrainer(object):
    def __init__(self, model, classifier, criterion, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.classifier = classifier.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion.to(self.device)
        # self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        ## record all the val
        record_json = {}
        not_improved_count = 0
        best_macro_epoch = 0
        best_micro_epoch = 0
        best_val_macro = 0
        best_val_micro = 0
        test_macro = 0
        test_micro = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            record_json[epoch] = log
            self._save_file(record_json)
            print(self.args.save_dir.split('/')[2])
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))
            if best_val_macro < log['val_macro']:
                best_val_macro = log['val_macro']
                test_macro = log['test_macro']
                best_macro_epoch = epoch
            if best_val_micro < log['val_micro']:
                best_val_micro = log['val_micro']
                test_micro = log['test_micro']
                best_micro_epoch = epoch
            print('Best results in validation set (macro):')
            print('best_macro_epoch: %d' % best_macro_epoch)
            print('val_macro_best: %.2f' % best_val_macro)
            print('test_macro_best: %.2f' % test_macro)

            print('Best results in validation set (micro):')
            print('best_micro_epoch: %d' % best_micro_epoch)
            print('val_micro_best: %.2f' % best_val_micro)
            print('test_micro_best: %.2f' % test_micro)
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

        self._save_file(record_json)

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

    def _save_file(self, log):
        if not os.path.exists(self.args.record_dir):
            os.mkdir(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name +'_'+self.args.save_dir.split('/')[2] +'.json')
        with open(record_path, 'w') as f:
            json.dump(log, f)

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

class Trainer(BaseTrainer):
    def __init__(self, model, classifier, criterion, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader,knowledge_dataloader):
        super(Trainer, self).__init__(model, classifier, criterion, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.knowledge_dataloader = knowledge_dataloader
        ## check the training
        self.writer = SummaryWriter()

    def _train_epoch(self, epoch):

        train_loss = 0
        print_loss = 0

        self.model.train()
        self.classifier.train()
        for batch_idx, (images, labels) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            if self.args.visual == 'BLIP':
                image_feature = self.model(images, None, mode='image')
                image_feature = image_feature[:, 0, :]    #[bs x 768]
                outputs = self.classifier(image_feature, images)

            elif self.args.visual == 'densenet':
                image_feature = self.model(None, images, mode='densenet')
                outputs = self.classifier(image_feature, images)

            elif self.args.visual == 'resnet101':
                _, image_feature = self.model(images)
                outputs = self.classifier(image_feature, images)

            loss = self.criterion(outputs, labels)
            train_loss += loss.item()
            self.writer.add_scalar("data/Loss", loss.item(), batch_idx + len(self.train_dataloader) * (epoch - 1))
            print_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if batch_idx % 5 == 0:
                print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, print_loss / 5))
                print_loss = 0

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print("Finish Epoch {} Training, Start Eval...".format(epoch))

        self.model.eval()
        self.classifier.eval()
        pre = []
        gt = []
        auc_pre = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_dataloader):
                images, labels = images.to(self.device), labels.float().to(self.device)

                if self.args.visual == 'BLIP':
                    image_feature = self.model(images, None, mode='image')
                    image_feature = image_feature[:, 0, :]  # [bs x 768]
                    outputs = self.classifier(image_feature, images)

            #=============================================================================
                elif self.args.visual == 'densenet':
                    image_feature = self.model(None, images, mode='densenet')
                    outputs = self.classifier(image_feature, images)

                elif self.args.visual == 'resnet101':
                    _, image_feature = self.model(images)
                    outputs = self.classifier(image_feature, images)
            # ============================================================================

                outputs_pred = outputs > 0.5
                outputs = outputs.float().cpu().numpy().tolist()
                outputs_pred = outputs_pred.float().cpu().numpy().tolist()
                auc_pre.extend(outputs)
                pre.extend(outputs_pred)
                gt.extend(labels.cpu().numpy().tolist())

        pre = np.array(pre)
        gt = np.array(gt)
        auc_pre = np.array(auc_pre)
        val_f1_macro = metrics.f1_score(gt, pre, average="macro")
        val_f1_micro = metrics.f1_score(gt, pre, average="micro")

        log.update({'val_macro': val_f1_macro})
        log.update({'val_micro': val_f1_micro})

        result = calculate_metrics(gt, pre, auc_pre, split='val')
        log.update(result)

        self.model.eval()
        self.classifier.eval()
        pre = []
        gt = []
        auc_pre = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_dataloader):
                images, labels = images.to(self.device), labels.float().to(self.device)

                if self.args.visual == 'BLIP':
                    image_feature = self.model(images, None, mode='image')
                    image_feature = image_feature[:, 0, :]  # [bs x 768]
                    outputs = self.classifier(image_feature, images)

            # =============================================================================
                elif self.args.visual == 'densenet':
                    image_feature = self.model(None, images, mode='densenet')
                    outputs = self.classifier(image_feature, images)

                elif self.args.visual == 'resnet101':
                    _, image_feature = self.model(images)
                    outputs = self.classifier(image_feature, images)
            # ============================================================================

                outputs_pred = outputs > 0.5
                outputs_pred = outputs_pred.float().cpu().numpy().tolist()
                outputs = outputs.float().cpu().numpy().tolist()
                pre.extend(outputs_pred)
                auc_pre.extend(outputs)
                gt.extend(labels.cpu().numpy().tolist())

        pre = np.array(pre)
        gt = np.array(gt)
        auc_pre = np.array(auc_pre)

        test_f1_macro = metrics.f1_score(gt, pre, average="macro")
        test_f1_micro = metrics.f1_score(gt, pre, average="micro")
        log.update({'test_macro': test_f1_macro})
        log.update({'test_micro': test_f1_micro})

        result = calculate_metrics(gt, pre, auc_pre, split='test')
        log.update(result)

        self.lr_scheduler.step()
        return log



