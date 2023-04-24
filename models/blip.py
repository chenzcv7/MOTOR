'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.vit_blip import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, AutoTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from medical_knowledge.SKG_knowledge import *
from models.tagencoder import TagEncoder

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config_blip.json',
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 args=None,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.args = args
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = 'configs/med_config_sci.json'
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.vision_proj = nn.Linear(vision_width, 256)
        self.text_proj = nn.Linear(768, 256)
        self.iu_proj = nn.Linear(768*2, 768)

        self.tag_encoder = TagEncoder(0.1, self.args)

    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"

        
        if mode=='image':
            if self.args.dataset_name == 'iu_xray':
                image_embeds0 = self.visual_encoder(image[:, 0])
                image_embeds1 = self.visual_encoder(image[:, 1])
                image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
                image_embeds = self.iu_proj(image_embeds)
            else:
                image_embeds = self.visual_encoder(image)

            return image_embeds
        
        elif mode=='text':
            # return text features
            text = self.tokenizer(caption, return_tensors="pt").to(image.device)
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            text = self.tokenizer(caption, return_tensors="pt").to(image.device)
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 args = None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        self.args = args
        self.prompt = prompt

        med_config = 'configs/med_config_sci.json'
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.vision_proj = nn.Linear(vision_width, 256)
        self.text_proj = nn.Linear(768, 256)
        if args.dataset_name == 'iu_xray':
            self.iu_proj = nn.Linear(768*2, 768)

    def forward(self, image, caption):
        if self.args.dataset_name == 'iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=90, return_tensors="pt").to(image.device)

        text.input_ids[:,0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)

        decoder_targets[:,:self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           labels = decoder_targets,
                                           return_dict = True,
                                          )


        loss_lm = decoder_output.loss

        return loss_lm
        
    def generate(self, image, sample=False, num_beams=3, max_length=90, min_length=10, top_p=0.9, repetition_penalty=1.0):
        if self.args.dataset_name == 'iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)

        knowledge_used = ''

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=1.1,
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])

        return captions, knowledge_used
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print(msg.missing_keys)
    return model        

def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    vision_width = 768
    visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                       num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                       drop_path_rate=0 or drop_path_rate
                                      )

    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                print(state_dict[key].shape)
                print(model.state_dict()[key].shape)
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
