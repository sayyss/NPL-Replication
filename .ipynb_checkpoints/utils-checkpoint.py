import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import re
import random

def prepare_text(filename):
    words = open(filename, "r").read()
    words = words.lower()
    words = re.sub(r'[^a-zA-Z\s]', '', words)
    words = words.split()
    
    # Create vocabulary with all unique words in text file
    vocab = sorted(list(set(words)))

    return words, vocab
    
def create_pairs(words, context_size):
    x = []
    y = []
    for i in range(len(words)-context_size):
        x.append(words[i:i+context_size])
        y.append(words[i+context_size:i+context_size+1])

    # x[i] -> ["asd","Asd","aw"] context_size=3
    # y[i] -> ["fgds"]
    return x,y


def get_index_vectors(x, y, words_to_i):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = words_to_i[x[i][j]]
        y[i] = words_to_i[y[i][0]]

    # x -> [12312,1231,1] context_size=3
    # y -> [5]
    return x,y

def get_word_dict(vocab):
    
    words_to_i = {}
    i_to_words = {}
    
    for i in enumerate(vocab):
        #('word', index) <=> (index, 'word')
        words_to_i[i[1]] = i[0]
        i_to_words[i[0]] = i[1]
        
    return words_to_i, i_to_words