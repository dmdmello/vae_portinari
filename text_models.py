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


class AE_LSTM(nn.Module):
    def __init__(self, nb_lstm_layers, nb_lstm_units=100, embedding_dim=300, nb_vocab_words = 3188, batch_size=32):
        super(AE_LSTM, self).__init__()

        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.nb_vocab_words = nb_vocab_words

        # when the model is bidirectional we double the output dimension
        self.lstm = nn.LSTM

        # build actual NN
        
        self.__build_model()

    def __build_model(self):

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        self.word_embedding = nn.Embedding(
            num_embeddings=self.nb_vocab_words,
            embedding_dim=self.embedding_dim
        )

        # design LSTM
        self.lstm_enc = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )
        
        self.lstm_dec = nn.LSTM(
            input_size=self.nb_lstm_units,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        '''
        if self.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()'''

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
    
    
    def encode(self, X, X_lengths):
        
        self.hidden_enc = self.init_hidden()

        batch_size, seq_len = X.size()
        #print(X.size())
        #print(batch_size)

        X = self.word_embedding(X)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        Z, self.hidden_enc = self.lstm_enc(X, self.hidden_enc)
        Z, _ = torch.nn.utils.rnn.pad_packed_sequence(Z, batch_first=True)
        
        timesteps = torch.tensor(Z.shape)[1]
        
        return Z[:,-1:,:].repeat((1,timesteps,1))

    
    def decode(self, Z, Z_lengths):
        
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden_dec = self.init_hidden()
        self.decoder_output = nn.Linear(self.nb_lstm_units, self.nb_vocab_words) 
        
        batch_size, seq_len, _ = Z.size()
        X_hat = torch.nn.utils.rnn.pack_padded_sequence(Z, Z_lengths, batch_first=True, enforce_sorted=False)

        print(batch_size)
        print(seq_len)
        
        X_hat, self.hidden_dec = self.lstm_dec(X_hat, self.hidden_dec)
  
        X_hat, _ = torch.nn.utils.rnn.pad_packed_sequence(X_hat, batch_first=True)

        X_hat = F.log_softmax(self.decoder_output(X_hat), dim = -1)

        return X_hat

