import json
import os
import random
import torch
import _pickle as cPickle

from torch.utils.data import Dataset,DataLoader

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
# from .utils import pre_caption
import os
from blip_original.utils import pre_question

from torchvision import transforms

nodes = 'noraml otherfinding heart cardiomegaly spine scoliosis pleural effusion thickening pneumothorax bone bonefractures lung emphysema pneumonia edema atelectasis cicatrix opacity lesion mediastinum hernia calcinosis foreignobject airspace airspacedisease hypoinflation'

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class VQAFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data', question_len=12, tokenizer=None):
        super(VQAFeatureDataset, self).__init__()
        self.args = args
        self.name = name
        assert name in ['train', 'test']
        if name == 'train':
            trainset_path = os.path.join(dataroot, 'trainset_kg_new.json')
        else:
            trainset_path = os.path.join(dataroot, 'testset_new.json')
        self.images_path = os.path.join(dataroot, 'images')
        self.trainset = json.load(open(trainset_path, 'rb'))

        self.dictionary = dictionary
        if self.args.maml:
            self.transform1 = transforms.Compose([transforms.Resize(84),
                                                 transforms.RandomCrop(84),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if self.args.autoencoder:
            self.transform2 = transforms.Compose([transforms.Resize(128),
                                                 transforms.RandomCrop(128),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.question_len = question_len

        self.num_ans_candidates = 487  # 56 431
        # close & open
        self.label2close = cPickle.load(open(os.path.join(dataroot, 'cache', 'close_label2ans.pkl'), 'rb'))
        self.label2open = cPickle.load(open(os.path.join(dataroot, 'cache', 'open_label2ans.pkl'), 'rb'))
        self.num_open_candidates = len(self.label2open)
        self.num_close_candidates = len(self.label2close)

        if args.autoencoder and args.maml:
            self.v_dim = args.v_dim * 2
        else:
            self.v_dim = args.v_dim

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, index):

        trainset = self.trainset[index]

        image = Image.open(os.path.join(self.images_path, trainset['image_name'])).convert('RGB')
        image_data = [0, 0]
        if self.args.maml:

            image_data[0] = self.transform1(image)
        if self.args.autoencoder:

            image_data[1] = self.transform2(image)

        question = pre_question(trainset['question'],max_ques_words=35)
        answer = trainset['answer']
        question_type = trainset['question_type']
        phrase_type = trainset['phrase_type']

        answer_type = trainset['answer_type']

        if answer_type == 'CLOSED':
            answer_target = 0
        else:
            answer_target = 1

        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            labels = torch.from_numpy(np.array(labels)).type(torch.int64)
            scores = torch.from_numpy(np.array(scores)).type(torch.float32)
            composed_target = torch.zeros(self.num_ans_candidates) # close + open
            if answer_target == 0:
                target = torch.zeros(self.num_close_candidates)
                if labels is not None:
                    target.scatter_(0, labels, scores)
                composed_target[:self.num_close_candidates] = target
            else:
                target = torch.zeros(self.num_open_candidates)
                if labels is not None:
                    target.scatter_(0, labels, scores)
                composed_target[self.num_close_candidates : self.num_ans_candidates] = target
            return image_data, question, composed_target, answer_type, question_type, phrase_type, answer_target
        else:

            return image_data, question, answer_type, question_type, phrase_type, answer_target


