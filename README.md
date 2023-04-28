# Towards Medical Artificial General Intelligence via Knowledge-Enhanced Multimodal Pretraining
<br>
The code of paper ["Towards Medical Artificial General Intelligence via Knowledge-Enhanced Multimodal Pretraining"](https://arxiv.org/abs/2304.14204) has been tested on PyTorch 1.10. To install the dependencies, run ["Adversarial Reinforced Instruction Attacker for Robust Vision-Language Navigation"](https://arxiv.org/abs/2107.11252)

```
pip install -r requirements.txt
```

# Dataset Installation
**MIMIC-CXR**: visit https://physionet.org/content/mimic-cxr/ to install <br>
**IU Xray**: visit https://openi.nlm.nih.gov/faq to install <br>
**VQA-RAD**: visit https://osf.io/89kps/ to install <br>
**SLAKE**: visit https://drive.google.com/file/d/1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U/view to install
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
