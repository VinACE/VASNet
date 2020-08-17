"""
# TODO Multihead implementation https://github.com/dreamgonfly/Transformer-pytorch/blob/master/models.py
https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/
https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

# Display attention mechanism.. shown in the above link for the paper..
"""

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, input_size=10, output_size=10 ): #  heads=8
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.m = input_size
        self.output_size = output_size

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by head size"

        self.values = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.keys = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False) 
        self.queries = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)        
        self.fc_out = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        apperture=-1
        ignore_itself=False
        self.apperture = apperture
        self.ignore_itself = ignore_itself

    def forward(self, x):
        # try:
        #     if query.shape[0] is not None:
        #         # import pdb;pdb.set_trace()

            N = x.shape[0]
            value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]

            # split embedding into self. head pieces
            values = values.reshape(N, value_len, self.heads, self.head_dim)
            keys = keys.reshape(N, key_len, self.heads, self.head_dim)
            queries = query.reshape(N, query_len, self.heads, self.head_dim)

            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(query)
            
            queries *= 0.06
            logits = torch.matmul(queries, keys.transpose(1,0))
            if self.ignore_itself:
                # Zero the diagonal activations (a distance of each frame with itself)
                logits[torch.eye(N).byte()] = -float("Inf")
            
            energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            # queries shape : (N, query_len, heads, heads_dim)
            # keyshape shape : (N, key_len, heads, heads_dim)
            # energy shape : (N, heads, query_len, key_len)

            if mask is not None:
                energy =  energy.masked_fill(mask == 0, float("-1e20")) # for numerical stability
            
            attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=-1) # Attention(Q,K,V) = sofmax(QK^{T}/(d_{k})**(1/2)) * V

            out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
                N, query_len, self.heads * self.head_dim
            )

            # Attention shape: (N, heads, query_len, key_len)
            # value shape: (N, Value_len, heads, heads_dim) key length and the value lenth are alwasy going to be the same.
            # after einsum (N, query_len, heads, head_dim) flatten last two dimension..
          
            weights = self.drop50(attention)

            out = self.fc_out(out)

            return out, weights
        # except:
        #     pass

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention, _ = self.attention(value, key, query, mask) # TODO check this place if weights _ is required are not..

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        yield out

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
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask,):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) ## Need to understand what is positions..
        
        for layers in self.layers:
            out = layers(out, out, out, mask, )
            
        return out # should we return the weights in Encoder.. or is it ok only to return on the Decoder part...
# DECODER LAYER..    
class DecoderBlock(nn.Module):         
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):  # x  & V & K are comming in from the encoder..
        attention, weights = self.attention(x, x, x, trg_mask)  # ENC (n x m) => (n x H)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out, weights

    
class Decoder(nn.Module):    ## DECODER BLOCK
    def __init__(
                    self, 
                    trg_vocab_size,
                    embed_size,
                    num_layers,
                    heads,
                    forward_expansion,
                    dropout,
                    device,
                    max_length,
                    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x, weights = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out, weights

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256, 
        num_layers=6, 
        forward_expansion=4,
        heads=8, 
        dropout=0, 
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size,
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size, 
            num_layers, 
            heads, 
            forward_expansion, 
            dropout, 
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask) # TODO weight should come here as well
        return out
    
if __name__ == "__main__":
    # import pdb;pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg =  torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0, 0,0], [1, 5, 6, 2, 4, 7, 6, 2,0,0]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
        device
    )
    out, weights = model(x, trg[:, :-1])
    # weight_mat_ = model(x, trg[:, :-1])[1]
    print(out)
    # print("**************")
    # print(weight_mat_)
