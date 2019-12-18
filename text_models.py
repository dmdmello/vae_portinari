import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from skimage.io import imread
from torchsummary import summary
import pandas as pd
import time
from text_models import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,
                 pre_trained_embedding: float,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=pre_trained_embedding)
        self.emb_dim = pre_trained_embedding.shape[1]
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
            
        self.rnn = nn.GRU(self.emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                enc_input: Tensor, 
                enc_input_len) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(enc_input))
        
        embedded_pack = torch.nn.utils.rnn.pack_padded_sequence(embedded, 
            enc_input_len, enforce_sorted=False)
        
        #print(embedded_pack)
        outputs_packed, hidden = self.rnn(embedded_pack)
        #print(outputs_packed[0].shape)
        #print(outputs_packed[1].shape)
        #print(outputs_packed.shape)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs_packed)
        #print(outputs.shape)
        
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        #print(hidden.shape)
        
        return outputs, hidden

    
class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 pre_trained_embedding: float,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=pre_trained_embedding)
        self.output_dim = pre_trained_embedding.shape[0]
        self.emb_dim = pre_trained_embedding.shape[1]
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.attention = attention
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + self.emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + self.emb_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)
        #print('a shape = {}'.format(a.shape))
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #print('encoder_outputs shape = {}'.format(encoder_outputs.shape))
        
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        #print('weighted_encoder_rep shape = {}'.format(weighted_encoder_rep.shape))
        
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        #print('weighted_encoder_rep shape = {}'.format(weighted_encoder_rep.shape))
        
        return weighted_encoder_rep


    def forward(self,
                dec_input: Tensor,
                #dec_input_len: Tensor, 
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        dec_input = dec_input.unsqueeze(0)

        embedded = self.dropout(self.embedding(dec_input))
        #print(embedded.shape)
        #print(f'embedded shape = {embedded.shape}')
        weighted_h = self._weighted_encoder_rep(decoder_hidden,
                                                encoder_outputs)
        
        #print(f'weighted_h shape = {weighted_h.shape}')
        rnn_input = torch.cat((embedded, weighted_h), dim = 2)
        
        #print(rnn_input.shape)
        #print(decoder_hidden.unsqueeze(0).shape)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_h = weighted_h.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_h,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)    

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                x: Tensor,
                x_l : Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = x.shape[1]
        max_len = x.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(x, x_l)

        # first input to the decoder is the <sos> token
        output = x[0,:]*0
        #print(f'first output shape = {output.shape}')
        
        for t in range(0, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (x[t,:] if teacher_force else top1)
            #if teacher_force: 
            #    print(f'output teacher force = {output}')
            
        return outputs
