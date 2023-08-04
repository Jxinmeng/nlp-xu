import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        src_att = self.multihead_attention(src, src, src, attn_mask=src_mask)[0]
        src_att = self.dropout1(src_att)
        src = self.layer_norm1(src + src_att)
        src_ffn = self.feed_forward(src)
        src_ffn = self.dropout2(src_ffn)
        src = self.layer_norm2(src + src_ffn)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multihead_attention = nn.MultiheadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.multihead_attention = nn.MultiheadAttention(d_model, n_head)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, src, tgt_mask, src_mask):
        tgt_att = self.masked_multihead_attention(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt_att = self.dropout1(tgt_att)
        tgt = self.layer_norm1(tgt + tgt_att)
        tgt_att = self.multihead_attention(tgt, src, src, attn_mask=src_mask)[0]
        tgt_att = self.dropout2(tgt_att)
        tgt = self.layer_norm2(tgt + tgt_att)
        tgt_ffn = self.feed_forward(tgt)
        tgt_ffn = self.dropout3(tgt_ffn)
        tgt = self.layer_norm3(tgt + tgt_ffn)
        return tgt

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_ffn, num_layers, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_head, d_ffn, dropout_rate) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_head, d_ffn, dropout_rate) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)

        # Encoder
        for encoder_layer in self.encoder:
            src_embedded = encoder_layer(src_embedded, src_mask)

        # Decoder
        for decoder_layer in self.decoder:
            tgt_embedded = decoder_layer(tgt_embedded, src_embedded, tgt_mask, src_mask)

        output = self.output_projection(tgt_embedded)
        return output
