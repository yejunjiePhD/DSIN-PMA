import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
# import numpy as np
from model.RevIN import RevIN
# from layers.Frft_encoder import frft_base, frft, FRFT
# from fracdiff2 import frac_diff_ffd
# from scipy.stats import pearsonr
#
# import numpy as np

# from layers.Transformer_encoder_similarity import TransformerEncoder
from layers.Transformer_encoder import TransformerEncoder


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.batch = configs.batch_size
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.multivariate = configs.enc_in
        self.alpha = configs.alpha


        self.output_attention = configs.output_attention

        ### Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)


        self.PositionalEmbedding = PositionalEmbedding(configs.d_model)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads, configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    configs.seq_len,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers_time)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        ####### 扩维 ## Embedding
        self.mapping = nn.Parameter(torch.randn(configs.seq_len, configs.d_model))
        self.value_embedding = nn.Linear(configs.seq_len, configs.d_model)

        self.dropout = nn.Dropout(configs.dropout )

        self.seq_pred = nn.Linear(self.seq_len * 16, configs.pred_len, bias=True)

        revin = True
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.multivariate, affine=True, subtract_last=False)

        self.flatten = nn.Flatten(start_dim=-2)
        self.Linear_concat = nn.Linear(configs.d_model * 1, 16)

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # self.TransformerEncoder_similarity_up = TransformerEncoder(configs.e_layers, configs.d_model, configs.n_heads, self.seq_len, configs.d_ff, configs.dropout)
        self.Transformer_encoder = TransformerEncoder(self.d_model, configs.n_heads, configs.e_layers_variate, configs.d_ff,
                                                      dropout=configs.dropout)


        self.FC = nn.Linear(configs.pred_len*2 , configs.pred_len)

        self.trend = nn.Linear(configs.seq_len, configs.pred_len)

    def Embedding(self, x_enc):
        enc_out = x_enc.transpose(2, 1).unsqueeze(-1)

        enc_out = enc_out * self.dropout(self.mapping)            # 32 7 96 128
        ## 位置嵌入 ####
        enc_out_p = enc_out.reshape(-1, self.seq_len, self.d_model)
        enc_out_p = self.PositionalEmbedding(enc_out_p) + enc_out_p     # 224 96 128
        enc_out = enc_out_p.reshape(-1, self.multivariate, self.seq_len, self.d_model)  # 32 7 96 128

        enc_out_in = enc_out.transpose(1, 0)            # 7 32 96 128
        enc_out = enc_out_in

        return enc_out_in, enc_out


    def Channel_independence(self, enc_out_in, trend_init):
        enc_out_in = enc_out_in.reshape(-1, self.seq_len, self.d_model)
        enc_out_in = self.encoder(enc_out_in, attn_mask=None)
        enc_out_in = enc_out_in.reshape(self.multivariate, -1, self.seq_len, self.d_model)  # 7 32 96 128

        enc_out = self.dropout(self.Linear_concat(enc_out_in))
        enc_out = enc_out.permute(1, 0, 2, 3)  # 32 7 96 128
        enc_out_in = self.flatten(enc_out)  # 32 7 96*16

        trend_init = self.dropout( self.trend(trend_init.transpose(2,1)) )

        return enc_out_in, trend_init

    def similarity_former(self, similarity_up_all, B, mask):
        similarity_up_all = similarity_up_all.reshape(B, -1, self.seq_len, self.d_model).transpose(2, 1)        # batch 96 7 128
        similarity_up_all = similarity_up_all.reshape(B, similarity_up_all.size(2) * self.seq_len, self.d_model)

        enc_out_state = self.TransformerEncoder_similarity_up(similarity_up_all, mask)    # 32 11 128
        enc_out_state = enc_out_state.reshape(B, self.seq_len, -1, self.d_model).permute(2, 0, 1, 3)

        return enc_out_state

    def correlation_matrix(self, x):
        # x batch 96 7
        # batch, input_len, N = x.shape

        reshaped_data = x.transpose(2,1)[-1]

        # 计算均值
        mean = reshaped_data.mean(dim=1, keepdim=True)
        # 中心化矩阵（减去均值）
        centered_matrix = reshaped_data - mean
        # 计算协方差矩阵
        cov_matrix = (centered_matrix @ centered_matrix.t()) / (reshaped_data.size(1) - 1)
        # 计算标准差矩阵
        std_dev = torch.sqrt(torch.diag(cov_matrix))
        # 广播标准差以准备计算相关系数
        std_dev_broadcast = std_dev.unsqueeze(1) * std_dev.unsqueeze(0)
        # 避免除以零（当标准差为0时）
        std_dev_broadcast[std_dev_broadcast == 0] = 1
        # 计算相关系数矩阵
        correlation_matrix = cov_matrix / std_dev_broadcast
        # 由于相关系数矩阵是对称的，我们只计算了上半部分
        # 填充下半部分为对称值
        correlation_matrix = (correlation_matrix + correlation_matrix.t()) / 2
        # 确保对角线为1（因为变量与自身的相关系数为1）
        correlation_matrix.fill_diagonal_(1)
        normalized_tensor = correlation_matrix

        correlation_matrix_mask = normalized_tensor < self.alpha  # N * N

        return correlation_matrix_mask.cuda()


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):


        x_enc = self.revin_layer(x_enc, 'norm')


        # x_enc_split = x_enc
        B, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # correlation_matrix
        mask = self.correlation_matrix(x_enc)

        # ######## 嵌入 ### 也可以考虑patch, 32 7 42 16, 然后16映射到128  self-attention
        ######### 分解  ###
        seasonal_init, trend_init = self.decompsition(x_enc)   # batch input_len variate
        # x_enc_time = seasonal_init

        enc_out_in, enc_out = self.Embedding(seasonal_init)         # 7 32 96 128
        enc_out_in, trend_init = self.Channel_independence(enc_out_in, trend_init)
        dec_out_time = self.seq_pred(enc_out_in).permute(0, 2, 1)   # 32 192 7
        dec_out_time = dec_out_time + trend_init.transpose(2,1)


        ########### similarity ###########
        enc_out_vari = self.value_embedding(x_enc.transpose(2,1))
        enc_out_vari = self.Transformer_encoder(enc_out_vari, mask)
        dec_out_vari = self.dropout(self.projector(enc_out_vari))


        ######### concat 融合  ##############
        enc_out_concat = torch.cat((dec_out_time.transpose(2,1), dec_out_vari), dim=-1)    # 7 32 96 256
        dec_out = self.FC(enc_out_concat)
        dec_out_ = self.revin_layer(dec_out.transpose(2,1), 'denorm')


        dec_out_time = self.revin_layer(dec_out_time, 'denorm')
        dec_out_vari = self.revin_layer(dec_out_vari.transpose(2,1), 'denorm')
        # dec_out_ = 0.5 * dec_out_vari + 0.5 * dec_out_time

        return dec_out_, dec_out_time, dec_out_vari


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, dec_out_time, dec_out_vari = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], dec_out_time, dec_out_vari    # [B, L, D]
