# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         classifier
# Description:  This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
# Author:       Boliu.Kelvin
# Date:         2020/4/7
#-------------------------------------------------------------------------------

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args): #hid_dim=2048
        super(SimpleClassifier, self).__init__()
        activation_dict = {'relu': nn.ReLU(inplace=True)}
        try:
            activation_func = activation_dict[args.activation]
        except:
            raise AssertionError(args.activation + " is not supported yet!")
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            activation_func,
            nn.Dropout(args.dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        #print(in_dim)
        self.main = nn.Sequential(*layers)
        self.projection = nn.Linear(768, 1024)
        self.args = args
    def forward(self, x):
        #print(x.size())
        if self.args.test_visual:
            logits = self.main(x)
        elif self.args.test_text:
            logits = self.main(x)
        elif self.args.test_all:
            logits = self.main(x)
        elif self.args.test_A:
            logits = self.main(x)
        elif self.args.test_B:
            logits = self.main(x)
        elif self.args.test_C:
            logits = self.main(x)
        else:
            x = self.projection(x)
            logits = self.main(x)
        return logits
