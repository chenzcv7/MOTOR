import json
import os
import csv
import random


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


if __name__ == "__main__":
    # label_list=[]
    # filename1 = '/home/mmvg/zicong/CX14/Data_Entry_2017_v2020.csv'
    # with open(filename1) as f1:
    #     reader1 = csv.reader(f1)
    #     reader1=list(reader1)
    #     # print(reader1[4]['Consolidation']=='1.0')
    #     f1.close()
    # print(reader1[1][1])
    # print(len(reader1))
    # train_val = []
    # test_list=[]
    # for line in open('/home/mmvg/zicong/CX14/train_val_list.txt',"r"):
    #     train_val.append(line.strip())
    # #tran:77872      val:8652    test:25596
    #
    # for line1 in open('/home/mmvg/zicong/CX14/test_list.txt',"r"):
    #     test_list.append(line1.strip())
    #
    # print(train_val[1])
    # print(train_val[0]==reader1[1][0])
    # val_list, train_list = data_split(train_val, ratio=0.1, shuffle=True)
    #
    # print(reader1[1][0])
    #
    # item=[]
    # label_list=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia','No Finding']
    # a = (0, 0, 0, 0)
    # b = ('image_path', 'report', 'label_index', 'split')
    # d = dict(zip(b, a))
    # for i in range(1,len(reader1)):
    #     label_index = [0 for j in range(15)]
    #     image_path=reader1[i][0]
    #     d['image_path']=image_path
    #     d['report']=[]
    #     cur_label=reader1[i][1].split('|')
    #     for label in cur_label:
    #         if label in label_list:
    #             label_index[label_list.index(label)] = 1
    #     d['label_index'] = label_index
    #     if image_path in train_list:
    #         d['split']='train'
    #     elif image_path in val_list:
    #         d['split']='val'
    #     elif image_path in test_list:
    #         d['split']='test'
    #     item.append(d.copy())
    #
    # with open('/home/mmvg/zicong/CX14/chexpert_dia.json', 'w') as file:
    #     file.write(json.dumps(item))

    # with open('/home/mmvg/zicong/CX14/chexpert_dia.json', encoding='utf-8') as f:
    #     load_dict = json.load(f)
    #     file_length = len(load_dict)
    #     f.close()
    #
    # print(load_dict[2])
    # print(load_dict[888])
    #
    # a = (0, 0, 0)
    # b = ('train', 'val', 'test')
    # d = dict(zip(b, a))
    # item = []
    # train = []
    # val = []
    # test = []
    # for i in range(file_length):
    #     if load_dict[i]['split'] == 'train':
    #         train.append(load_dict[i])
    #     elif load_dict[i]['split'] == 'val':
    #         val.append(load_dict[i])
    #     elif load_dict[i]['split'] == 'test':
    #         test.append(load_dict[i])
    #
    # d['train'] = train
    # d['val'] = val
    # d['test'] = test
    # item.append(d.copy())
    # with open('/home/mmvg/zicong/CX14/chexpert_diagnosis.json', 'w') as file:
    #     file.write(json.dumps(item))
    annotation = json.load(open(os.path.join('/home/mmvg/zicong/MIMIC-CXR/mimic_cxr/annotation_kg.json'), 'r'))
    ann_10_list, _ = data_split(annotation, ratio=0.5, shuffle=True)
    print(len(annotation))
    print(len(ann_10_list))
    with open('/home/mmvg/zicong/MIMIC-CXR/mimic_cxr/annotation_50.json', 'w') as file:
        file.write(json.dumps(ann_10_list))

