import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
# from models.blip import init_tokenizer
from models.med import BertConfig,BertModel
from transformers import BertTokenizer, AutoTokenizer

nodes = [
    'normal','other finding','heart','cardiomegaly','spine','scoliosis','pleural','effusion','thickening','pneumothorax',
    'bone', 'bone fractures','lung','emphysema,','pneumonia','edema','atelectasis','clcatrix','opacity','lesion',
    'mediastinum','hernia','calcinosis','foreign object','airspace','airspace disease','hypoinflation'
]

node_inds = [0,1,2,3,3,4,4,5,5,5,5,6,6,7,7,7,7,7,7,7,7,8,8,8,9,10,10,10]
node_labels = [0,2,3,1,4,1,5,1,6,6,6,1,7,1,8,8,8,8,8,8,8,1,9,9,10,1,11,11]

def init_tokenizer(args):
    if args.bert == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.bert == 'sci':
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    elif args.bert == 'cli':
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model,self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(20, d_model)
        position = torch.arange(0, 20).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.inds = [0,1,2,3,3,4,4,5,5,5,5,6,6,7,7,7,7,7,7,7,7,8,8,8,9,10,10,10]

    def forward(self, x, device):
        tmp_pe = self.pe[:, :x.size(1)]
        final_pe = torch.zeros(x.size())
        for num in range(len(self.inds)):
            final_pe[:,num] = tmp_pe[:,self.inds[num]]
        x = x + final_pe.to(device)
        return self.dropout(x)


def generate_mask(inds,labels, x):
    nbatches = x.size(0)
    mask = torch.zeros([nbatches,len(inds),len(inds)])
    for i in range(len(inds)):
        for j in range(len(inds)):
            if i == 0 or j ==0:
                mask[:,i,j]=1
            if labels[i] == labels[j] or inds[i]==inds[j]:
                mask[:,i,j]=1
                
    return mask.to(x.device)

	

class TagEncoder(nn.Module):

    def __init__(self,dropout, args):
        super(TagEncoder,self).__init__()
        c = copy.deepcopy
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(EncoderLayer(768, c(MultiHeadedAttention(6, 768)), c(PositionwiseFeedForward(768, 1024, 0.1)), dropout), 2)
        # self.embedds = Embeddings(27,768)
        self.pe = PositionalEncoding(768, dropout)
        self.tokenizer = init_tokenizer(args)
        med_config = 'configs/tag_config_sci.json'
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = 768
        if args.task == 'pretrain':
            self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased',config=encoder_config, add_pooling_layer=False, SKG_know=True)
            self.bert.resize_token_embeddings(len(self.tokenizer))
        else:
            self.bert = BertModel(config=BertConfig.from_json_file('configs/tag_config_sci_down.json'), add_pooling_layer=False, SKG_know=True)

    def forward(self,x, device):
        x = self.get_tag_embeds(x, device)
        x = self.pe(x, device)
        x = self.bert.embeddings.LayerNorm(x)
        x= self.bert.embeddings.dropout(x)
        x = self.encoder(x,generate_mask(node_inds,node_labels, x))
        return self.dropout(x)

    def get_tag_embeds(self,x, device):
        # x = ' '.join(x)
        # return ids
        x = self.tokenizer(x, padding='max_length', truncation=True, max_length=32,
                              return_tensors="pt").to(device)
        # to get the word embeddings
        x = self.bert.embeddings.word_embeddings(x.input_ids)
        # get the average of two tokens
        x[:,2,:] = (x[:,2,:]+x[:,3,:])/2
        x[:,13,:] = (x[:,13,:]+x[:,14,:])/2
        x[:,26,:] = (x[:,26,:]+x[:,27,:])/2
        x[:,29,:] = (x[:,29,:]+x[:,30,:])/2
        # delete the others
        x = torch.cat((x[:,:3,:],x[:,4:14,:],x[:,15:27,:],x[:,28:30,:],x[:,31:,:]),dim=1)
        return x
		
### when using

# te = TagEncoder(0.1)
#
# tag_features = te(nodes)