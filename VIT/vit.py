from curses.ascii import FF
from turtle import forward
import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, activation = 'gelu'):
        super().__init__()
        if activation == 'gelu':
            activation = nn.GELU
        else:
            activation = nn.ReLU
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, nheads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()

        self.nheads = nheads
        self.dim_head = dim_head
        self.dimD = nheads * dim_head
        self.z = dim_head ** -0.5

        self.dropout = nn.Dropou(dropout)

        self.Mq = nn.Linear(dim, self.dimD, bias=False)
        self.Mk = nn.Linear(dim, self.dimD, bias=False)
        self.Mv = nn.Linear(dim, self.dimD, bias=False)

        self.score = nn.Softmax(dim=-1)

        self.out = nn.Sequential(
            nn.Linear(self.dimD, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        bs, n, d = x.shape
        q = self.Mq(x).view(bs, -1, self.nheads, self.dim_head).transpose(1,2) #(bs, nheads, q_length, dim_head)
        k = self.Mk(x).view(bs, -1, self.nheads, self.dim_head).transpose(1,2) #(bs, nheads, q_length, dim_head)
        v = self.Mv(x).view(bs, -1, self.nheads, self.dim_head).transpose(1,2) #(bs, nheads, q_length, dim_head)

        q = q * self.z

        weights = self.score(torch.matmul(q,k.transpose(2,3)))
        weights = self.dropout(weights)

        context = torch.matmul(weights,v)
        context = context.transpose(1,2).contiguous().view(bs, -1, self.nheads * self.dim_head)

        return self.out(context)

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nheads, dim_head, dropout, activation):
        super().__init__()
        self.attnBlock = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadSelfAttention(dim, nheads=nheads, dim_head=dim_head, dropout=dropout)
        )

        self.ffnBlock = nn.Sequential(
            nn.LayerNorm(dim),
            FFN(dim, hidden_dim=hidden_dim, dropout=dropout, activation=activation)
        )

    def forward(self, x):
        attn_out =  self.attnBlock(x)
        attn_out = attn_out + x

        ffn_out = self.ffnBlock(attn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out

class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, nheads, dim_head, dropout, activation):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, hidden_dim, nheads, dim_head, dropout, activation))

    def forward(self, x):
        return self.layers(x)

class Vit(nn.Module):
    def __init__(self,
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    hidden_dim,
    nheads=8,
    dim_head=64,
    dropout=0.1,
    activation='gelu',
    channels=3,
    pool='cls'):
        super().__init__()
        IH, IW = image_size
        self.PH, self.PW = patch_size

        assert IH % self.PH == 0 and IW % self.PW == 0

        self.nPatches = (IH//self.PH) * (IW//self.PW)
        self.dim_patch = channels * self.PH * self.PW

        self.to_emb = nn.Linear(self.dim_patch, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.nPatches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(depth, dim, hidden_dim, nheads, dim_head, dropout, activation)

        self.pool = pool
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        bs, c, h, w = img.shape
        inp = img.reshape(bs, c, h//self.PH, self.PH,  w//self.PW, self.PW).permute(0,2,4,3,5,1).reshape(bs, self.nPatches, self.dim_patch )

        cls_token = self.cls_token.repeat(bs, 0)
        inp = torch.cat((cls_token, inp), dim=1)
        inp += self.pos_embedding[:, :(h*w + 1)]
        inp = self.dropout(inp)

        out = self.transformer(inp)

        if self.pool == 'mean':
            out = out.mean(dim=1)
        else:
            out = out[:, 0]

        return self.head(out)