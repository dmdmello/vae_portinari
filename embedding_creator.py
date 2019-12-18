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
import unidecode

def preprocess(load = False, embedding_source = 'glove.6B.300d.NILC.txt', 
               embedding_portinari = '../data/data-portinari/embedding_matrix_NILC.npy', remove_accents = False):
    flag=0
    i = 0
    last_count = 0
    new_count = 0
    lines = []
    PATH_PORTINARI = "../data/data-portinari/portinari_200x200/"

    portinari_desc = []
    retratos_ordem = []
    resto_ordem = []

    with open(PATH_PORTINARI + "/tb_descricao_sem_crase.txt", encoding='latin-1') as openfile:
        for line in openfile:
            if remove_accents:
                line = unidecode.unidecode(line)

            line_without_number = line.split()[1:]
            string_accumulator = ''
            for word in line_without_number:
                string_accumulator = string_accumulator + ' ' + word
            portinari_desc.append(string_accumulator)            

            split = line.split()
            for part in split[-10:]:
                if "Retrato" in part:
                    flag = 1
                elif "de" in part and (flag==1):
                    new_count = last_count + 1
                    flag = 0
                    line_without_number = line.split()[1:]
                    lines.append(line)
                    #string_accumulator = ''
                    #for word in line_without_number:
                    #    string_accumulator = string_accumulator + ' ' + word
                    retratos_ordem.append(i)
                    #retratos_desc.append(line_without_number)
                else:
                    #resto_ordem.append(i)
                    flag = 0
            if new_count == last_count:
                resto_ordem.append(i)
            last_count = new_count
            i = i+1                
    print('Achou {} retratos'.format(last_count))

    
    MAX_NB_WORDS = 50000
    MAX_LEN = 200
    #print(retratos_desc)
    tokenize = Tokenizer(num_words = MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ \t\n'+"'")
    tokenize.fit_on_texts(portinari_desc)
    portinari_idx = tokenize.texts_to_sequences(portinari_desc)
    word_index = tokenize.word_index
    #adds tokens for EOS and SOS
    word_index['<unk>'] = len(word_index)+1
    word_index['<sos>'] = len(word_index)+1
    word_index['<eos>'] = len(word_index)+1
    
    portinari_idx = [sent[:MAX_LEN] for sent in portinari_idx]

    portinari_idx = [[word_index['<sos>']] + sent + [word_index['<eos>']] for sent in portinari_idx]
    
    retratos_idx = [portinari_idx[i] for i in retratos_ordem]
    resto_idx = [portinari_idx[i] for i in resto_ordem]
    
    index_word = {v: k for k, v in word_index.items()}
    key_list = list(word_index.keys())
    val_list = list(word_index.values())


    GLOVE_DIR = "../data/glove/"
    EMBEDDING_DIM = 300
    MAX_NB_WORDS = 50000000
    
    if(load):
        try:
            embedding_matrix = np.load('../data/data-portinari/embedding_matrix.npy')
        except:
            print('embedding not found!')
    else:
        print('Indexing word vectors.')
        embeddings_index = {}
        f = codecs.open(os.path.join(GLOVE_DIR, embedding_source), encoding='utf-8')
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        
        nb_words = min(MAX_NB_WORDS, len(word_index))
        
        ADD_TOKENS = 2 # number of word marks not present in the pre trained embeddings
        embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM+ADD_TOKENS))
        embedding_matrix[-3] = np.append(embeddings_index.get('<unk>'), ADD_TOKENS*[0])
        embedding_matrix[-2, -2] = 1.0 #<sos>
        embedding_matrix[-1, -1] = 1.0 #<eos>
        
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            if word not in ['<sos>', '<eos>']:
                
                embedding_vector = embeddings_index.get(word)
             
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = np.append(embedding_vector, ADD_TOKENS*[0])
                else:
                    embedding_matrix[i] = embedding_matrix[-3]
                    #embedding_matrix[i] = embedding_matrix[0]
                    #unkown_tokens.append(i)
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        
        np.save('../data/data-portinari/embedding_matrix',embedding_matrix)
    
    emb_length = len(embedding_matrix)


    unknown_tokens = [i for i in range(len(embedding_matrix)) if np.sum(embedding_matrix[i]-embedding_matrix[-3]) == 0]
    for tk in unknown_tokens:
        index_word[tk] = '<unk>'

    null_words = []
    total_words = []
    
    for i in portinari_idx:
        for j in i:
            total_words.append(j)
            if index_word[j] == '<unk>':
                null_words.append(j)

    print(f'total words {len(total_words)}')
    print(f'null words {len(null_words)}')
    print('{:.2f} % de palavras não encontradas no emb'.format(100*(len(null_words)/len(total_words))))

    zero_axis_emb = np.sum(embedding_matrix, axis=1) == 0
    zero_axis = [i for i in range(len(zero_axis_emb)) if zero_axis_emb[i] == True]
    
    
    return embedding_matrix, portinari_idx, retratos_idx, resto_idx, word_index, index_word

