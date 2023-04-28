# Towards Medical Artificial General Intelligence via Knowledge-Enhanced Multimodal Pretraining
<br>

Code of our paper "Towards Medical Artificial General Intelligence via Knowledge-Enhanced Multimodal Pretraining" [[Paper]](https://arxiv.org/abs/2304.14204)

# Environment Installation
Our code has been tested on PyTorch 1.10. To install the dependencies, run

```
pip install -r requirements.txt
```


# Dataset
**MIMIC-CXR**: Download from https://physionet.org/content/mimic-cxr/ <br>
**IU Xray**: Download from https://openi.nlm.nih.gov/faq <br>
**Chexpert 14**: Download from https://nihcc.app.box.com/v/ChestXray-NIHCC <br>
**VQA-RAD**: Download from https://osf.io/89kps/ <br>
**SLAKE**: Download from https://drive.google.com/file/d/1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U/view 
# Pretrain
```
python -m torch.distributed.run --nproc_per_node=2 Pretrain.py  --output_dir ./output/Pretrain
```
# Downstream Tasks
## Report Generation
IU Xray
```
python Generation_BLIP.py --output_dir output/Generation --dataset_name iu_xray --pretrained [Pretrained checkpoint] --distributed False --save_dir results/iu/Generation
```
## Diagnosis Classfication
MIMIC-CXR
```
python Diagnose_BLIP.py --output_dir output/Diagnose --dataset_name mimic_cxr --pretrained [Pretrained checkpoint] --distributed False --save_dir results/mimic/Diagnose
```
Chexpert 14
```
python Diagnose_BLIP.py --output_dir output/Diagnose --dataset_name chexpert --pretrained [Pretrained checkpoint] --distributed False --save_dir results/chexpert/Diagnose
```
## Image-Report Retrieval
MIMIC-CXR
```
python -m torch.distributed.run --nproc_per_node=1 Retrieval_BLIP.py --output_dir output/Retrieval_mimic --dataset_name mimic_cxr --pretrained [Pretrained checkpoint]  --record_dir results/mimic/Retrieval
```
## Medical VQA
VQA-RAD
```
python VQA.py --test_C --add_typeatt2 --pretrained [Pretrained checkpoint] --setting VQA-RAD
```
SLAKE
```
python VQA_slake.py --test_C --add_typeatt2 --pretrained [Pretrained checkpoint] --setting VQA-SLAKE
```

# Citation

If you find this repository is useful, please consider citing our paper:

```
@article{Lin2023TowardsMA,
      title={Towards Medical Artificial General Intelligence via Knowledge-Enhanced Multimodal Pretraining}, 
      author={Bingqian Lin and Zicong Chen and Mingjie Li and Haokun Lin and Hang Xu and Yi Zhu and Jianzhuang Liu and Wenjia Cai and Lei Yang and Shen Zhao and Chenfei Wu and Ling Chen and Xiaojun Chang and Yi Yang and Lei Xing and Xiaodan Liang},
      journal={arXiv preprint arXiv:2304.14204},
      year={2023}
}
```
