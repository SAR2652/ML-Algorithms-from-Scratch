import sys
import torch

sys.path.append('../')
from PyTorch.RNN import RNNLayer


sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
print(sentence_int)

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

# print(embedded_sentence)
print(f'Embedded sentence shape = {embedded_sentence.shape}')

rnn = RNNLayer(embedded_sentence.size()[-1], 20)
h_out = rnn(embedded_sentence)

# print(h_out)
print(h_out.shape)
