import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_size, heads, dropout, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_id = num_layers
        # self.max_length = 336

    def forward(self, value, mask):
        key = query = value
        N = query.shape[0]

        x = query

        for layer in self.layers:
            x = layer(x, x, x, mask)      # 传入 V K Q

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, d_ff):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttentionLayer(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, value, key, query, mask):

        attention = self.attention(query, query, query, mask)

        # Add skip connection and run through normalization
        x = self.norm1(attention + query)

        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(SelfAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.keys = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.queries = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask=None):

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.dropout(self.values(values))
        keys = self.dropout(self.keys(keys))
        query = self.dropout(self.queries(query))

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 1, float("-1e-10"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)


        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.dropout(self.fc_out(out))
        return out
