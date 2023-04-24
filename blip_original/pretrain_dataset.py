import json
import os
import random
import torch

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import pre_caption
import os
import numpy as np

label_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
              'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
              'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

node = [
    'normal','other finding','heart','cardiomegaly','spine','scoliosis','pleural','effusion','thickening','pneumothorax',
    'bone', 'bone fractures','lung','emphysema','pneumonia','edema','atelectasis','clcatrix','opacity','lesion',
    'mediastinum','hernia','calcinosis','foreign object','airspace','airspace disease','hypoinflation'
]
nodes = ' '.join(node)


class pretrain_dataset(Dataset):
    def __init__(self, ann_path_all, image_dir_all, transform, max_words=90, args=None):

        self.ann_path = ann_path_all

        with open(self.ann_path, encoding='utf-8') as f:
            self.ann_all = json.load(f)
            f.close()

        self.ann = self.ann_all[0]['train']

        self.transform = transform
        self.max_words = max_words
        self.mimic_image_dir = image_dir_all
        self.args = args

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = ann['image_path']

        #mimic_cxr
        image = Image.open(os.path.join(self.mimic_image_dir, image_path[0])).convert('RGB')
        image = self.transform(image)

        knowledge_skg = nodes

        triplet_len = len(ann['triplet'])
        if triplet_len > 30:
            triplets = ann['triplet'][:30]
            knowledge_tc = '-'.join(triplets)
        else:
            knowledge_tc = '-'.join(ann['triplet'])

        knowledge_tc = pre_caption(knowledge_tc, self.max_words)  #triplet can see each other

        report = pre_caption(ann['report'], self.max_words)

        label = np.array(ann['label_index'])

        return image, report, knowledge_skg, knowledge_tc, label