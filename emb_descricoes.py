#!/usr/bin/env python
# coding: utf-8

# In[24]:


import shutil
import errno
import zipfile
import os
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import codecs
from keras.preprocessing.text import Tokenizer


def preprocess():
    flag=0
    i = 0
    count = 0
    lines = []
    PATH_PORTINARI = "../data/data-portinari/portinari_200x200/"
    
    portinari_desc = []
    retratos_ordem = []
    retratos_desc = []

    with open(PATH_PORTINARI + "/tb_descricao_sem_crase.txt", encoding='latin-1') as openfile:
        for line in openfile:
            portinari_desc.append(line)
            i = i+1
            split = line.split()
            for part in split[-10:]:
                if "Retrato" in part:
                    flag = 1
                elif "de" in part and (flag==1):
                    count = count + 1
                    flag = 0
                    lines.append(line.split())
                    retratos_ordem.append(i)
                    retratos_desc.append(line)
                else:
                    flag = 0
    print('Achou {} retratos'.format(count))

    retratos_idx_img = [int(l[0]) for l in lines]
    '''
    for i in range(len(retratos_desc)):
        for j in range(len(retratos_desc[i])):
            if retratos_desc[i][j] == 'à':
                #retratos_desc[i][j] = 'a'
                print(retratos_desc[i][j])
            if retratos_desc[i][j] == 'às':
                retratos_desc[i][j] = 'as'
                print(retratos_desc[i][j])
            else:
                continue'''
    
    MAX_NB_WORDS = 50000
    tokenize = Tokenizer(nb_words = MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ \t\n')
    tokenize.fit_on_texts(retratos_desc)
    retratos_idx = tokenize.texts_to_sequences(retratos_desc)
    word_index = tokenize.word_index
    index_word = {v: k for k, v in word_index.items()}

    key_list = list(word_index.keys())
    val_list = list(word_index.values())

    first_tokens = [sq[0] for sq in retratos_idx]
    first_words = [index_word[x[0]] for x in retratos_idx]

    GLOVE_DIR = "../data/glove/"
    EMBEDDING_DIM = 300
    MAX_NB_WORDS = 50000000

    '''print('Indexing word vectors.')
    embeddings_index = {}
    f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
    emb_words = []
    for line in f:
        values = line.split(' ')
        word = values[0]
        emb_words.append(word)
    f.close()

    nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    np.save('../data/data-celeba/embedding_matrix',embedding_matrix)'''
    embedding_matrix = np.load('../data/data-celeba/embedding_matrix.npy')

    zero_axis_emb = np.sum(embedding_matrix, axis=1) == 0
    zero_axis = [i for i in range(len(zero_axis_emb)) if zero_axis_emb[i] == True]

    null_words = []
    total_words = []
    for i in retratos_idx:
        for j in i:
            total_words.append(j)
            if zero_axis_emb[j]:
                null_words.append(j)
    retratos_len = [len(i) for i in retratos_desc]
    print('{:.2f} % de palavras não encontradas no emb'.format(100*(len(null_words)/len(total_words))))

    zero_axis_emb = np.sum(embedding_matrix, axis=1) == 0
    zero_axis = [i for i in range(len(zero_axis_emb)) if zero_axis_emb[i] == True]
    
    return embedding_matrix, retratos_idx, word_index, index_word

