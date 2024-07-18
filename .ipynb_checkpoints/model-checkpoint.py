import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import re
import random
from utils import prepare_text, create_pairs, get_index_vectors, get_word_dict

# Model
"""
y = b + Wx + U * tanh(d + Hx)

x = concat of all input sequence feature vectors(words)
b = biases for W
d = biases for H
W = direct representation matrix
H = hidden layer matrix
U = another hidden to output layer matrix

y = (Wx + b) + (U * tanh(d+Hx))
y =  (1,|V|) +   (1, |V|) 
     
goes to two different models, addition = (1,|V|) + (1, |V|) = (1,|V|)
|V| -> length of vocabuluary

then (1,|V|) -> softmax -> probabilities for each word in vocab
"""


class NPL:

    def __init__(self, vocab, hidden_units=100, context_size=3, feature_word_len=10, has_direct_rep=True):
        
        self.hidden_units = hidden_units
        self.feature_word_len = feature_word_len
        self.has_direct_rep = has_direct_rep
        self.context_size = context_size
        self.vocab = vocab


        self.C = torch.randn(len(self.vocab), feature_word_len)
        self.hidden_layer = torch.randn((self.context_size*self.feature_word_len), self.hidden_units)
        self.b = torch.randn(self.hidden_units)
        self.output_layer = torch.randn(self.hidden_units, len(self.vocab))
        
        self.parameters = [self.C, self.hidden_layer, self.b, self.output_layer]
        
        if has_direct_rep:
            self.direct_representation = torch.randn((self.context_size*self.feature_word_len), len(self.vocab))
            self.d = torch.randn(len(self.vocab))
            self.parameters.extend([self.direct_representation, self.d])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.CLE = nn.CrossEntropyLoss()

        # Set parameters gradient to true
        for p in self.parameters:
            p.requires_grad = True
            
    # List of word indexes to feature vectors
    def get_feature_vectors(self, x):

        # C[[index_1,index_2,index_3],...]
        x = self.C[x]
       
        # concat all input feature vectors into one
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]) # [B, context_size*feature_vector_len)
        
        return x
        
    def forward(self, x):

        x = self.get_feature_vectors(x)
        #print(x.shape,y.shape) # [B, context_size*feature_vector_len] , [B, feature_vector_len]
        
        # Hidden layer tanh(b+Hx)
        H = self.tanh(torch.matmul(x, self.hidden_layer) + self.b)
        O = torch.matmul(H, self.output_layer) # [B, |vocab|]

        if self.has_direct_rep:
            # Direct representation layer (Wx + d)
            D = torch.matmul(x, self.direct_representation) + self.d
            logits = O + D
        else:
            logits = O

        return logits
        
    def __call__(self, x):
        logits = self.forward(x)
        return logits
        
    def generate(self, start_context, length):

        if type(start_context) is not str:
            raise "Context has to be a string"        

        start_context = start_context.split()

        if len(start_context) > self.context_size:
            print("input string larger than context size, might lead to improper responses\n")

        elif len(start_context) < self.context_size:
            raise f"Input needs to be atleast {self.context_size} words"

        generated_text = start_context
        current_context = start_context[-self.context_size:]
        
        for i in range(length):

            index_vectors = torch.tensor([[self.vocab.index(word) for word in current_context]])
            logits = self.forward(index_vectors)
            prob = self.softmax(logits)
            next_pred = self.vocab[torch.argmax(prob)]
            generated_text.append(next_pred)
            current_context = generated_text[-self.context_size:]
            
        return ' '.join(generated_text)


def train(text_file, **kwargs):
    
    defaults = {
        'hidden_units': 100,
        'context_size': 3,
        'feature_vector_size': 10,
        'direct_rep': False,
        'epochs': 50,
    }

    defaults.update(kwargs)

    # Prepare data
    words, vocab = prepare_text(text_file)

    # Helper dictionaries mapping words to index and vice versa
    words_to_i, i_to_words = get_word_dict(vocab)
    
    x,y = create_pairs(words, defaults['context_size'])
    x,y = get_index_vectors(x,y, words_to_i)

    x,y = torch.tensor(x), torch.tensor(y)
    # Model
    model = NPL(vocab=vocab, hidden_units=defaults['hidden_units'], context_size=defaults['context_size'], 
                feature_word_len=defaults['feature_vector_size'], has_direct_rep=defaults['direct_rep'])

    # optimizer and loss
    softmax = nn.Softmax(dim=1)
    CLE = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters, lr=0.01, momentum=0.9)

    for epoch in range(defaults['epochs']):

        # Random 50 indexes 
        res = random.sample(range(0, x.shape[0]), 50)
        batch_x = x[res]
        batch_y = y[res]

        logits = model(batch_x)
        loss = CLE(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model

