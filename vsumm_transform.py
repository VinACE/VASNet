import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads ): #  heads=8
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by head size"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) 
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)        
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N= query.shape[0]
        value_len, key_len, query_len = values.shape[1], key.shape[1], query.shape[1]

        # split embedding into self. head pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim) # TODO check this later for the  query in the the video

        energy = torch.einsum("nqhd,nkhd->nhqk" [queries,keys])
        # queries shape : (N, query_len, heads, heads_dim)
        # keyshape shape : (N, key_len, heads, heads_dim)
        # energy shape : (N, heads, query_len, key_len)

        if mask is not None:
            energy =  energy.masked_fill(mask =0, float("-1e20")) # for numerical stability
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=1024) # Attention(Q,K,V) = sofmax(QK^{T}/(d_{k})**(1/2)) * V

        out = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # Attention shape: (N, heads, query_len, key_len)
        # value shape: (N, Value_len, heads, heads_dim) key length and the value lenth are alwasy going to be the same.
        # after einsum (N, query_len, heads, head_dim) flatten last two dimension..

        out = self.fc_out(out)

        return out, attention

class TransformerBlock(nn.module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.normal1(attention + query))
        forward = self.feed_forward(x)
        out = self.droput(self.norm2(forward + x))
        return out, attention


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [ 
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion = forward_expansion
                )
            ]
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        

        







