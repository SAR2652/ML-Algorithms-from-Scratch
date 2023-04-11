import torch
from PyTorch.RNN import GRU

sentence = 'Life is short, eat dessert first'

dc = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
print(sentence_int)

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

# print(embedded_sentence)
print(f'Embedded sentence shape = {embedded_sentence.shape}')

embedding_shape = embedded_sentence.size()[-1]
gru = GRU(embedding_shape, 20)
h_out = gru(embedded_sentence)

print(f'Output Shape = {h_out.shape}')
