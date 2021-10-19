import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from torch.nn.modules import activation


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class TransposedConv1d(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=3,
                 stride=2,
                 padding=1,
                 output_padding=1,
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=True):
        super(TransposedConv1d, self).__init__()

        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn

        self.transposed_conv1d = nn.ConvTranspose1d(in_channels,
                                                    output_channels,
                                                    kernel_shape,
                                                    stride,
                                                    padding=padding,
                                                    output_padding=output_padding,
                                                    bias=use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.transposed_conv1d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class TransposedConv3d(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(3, 3, 3),
                 stride=(2, 1, 1),
                 padding=(1, 1, 1),
                 output_padding=(1, 0, 0),
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=True):
        super(TransposedConv3d, self).__init__()

        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn

        self.transposed_conv3d = nn.ConvTranspose3d(in_channels,
                                                    output_channels,
                                                    kernel_shape,
                                                    stride,
                                                    padding=padding,
                                                    output_padding=output_padding,
                                                    bias=use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.transposed_conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding='spatial_valid',
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=False):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.padding = padding

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        if self.padding == 'same':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        if self.padding == 'spatial_valid':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f

            pad = [0, 0, 0, 0, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class Unit1D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=1,
                 stride=1,
                 padding='same',
                 activation_fn=F.relu,
                 use_bias=True):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                output_channels,
                                kernel_shape,
                                stride,
                                padding=0,
                                bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - (t % self._stride), 0)

    def forward(self, x):
        if self._padding == 'same':
            batch, channel, t = x.size()
            pad_t = self.compute_pad(t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerHead(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 max_poslen=256,
                 nheads=8,
                 dropout=0.1,
                 nlayers=2,
                 activation_fn=F.relu
        ):
        super(TransformerHead, self).__init__()
        self.pos_encoder = PositionalEncoding(in_channels, dropout, max_len=max_poslen)
        self.layer_norm = LayerNorm(in_channels)
        encoder_layers = TransformerEncoderLayer(in_channels, nheads, int(in_channels / 2), dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = in_channels
        self.decoder = nn.Linear(in_channels, output_channels)
        self.activation_fn=activation_fn

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    # def init_weights(self, init_type='kaiming_normal'):
    #     """Initialize Transformer module.
    #     :param torch.nn.Module model: transformer instance
    #     :param str init_type: initialization type
    #     """
    #     # if init_type == "pytorch":
    #     #     return

    #     # weight init
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             if init_type == "xavier_uniform":
    #                 torch.nn.init.xavier_uniform_(p.data)
    #             elif init_type == "xavier_normal":
    #                 torch.nn.init.xavier_normal_(p.data)
    #             elif init_type == "kaiming_uniform":
    #                 torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
    #             elif init_type == "kaiming_normal":
    #                 torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
    #             else:
    #                 raise ValueError("Unknown initialization: " + init_type)
    #     # bias init
    #     for p in self.parameters():
    #         if p.dim() == 1:
    #             p.data.zero_()

    #     # reset some modules with default init
    #     for m in self.modules():
    #         if isinstance(m, (torch.nn.Embedding, LayerNorm)):
    #             m.reset_parameters()
                

    def forward(self, src):
        """ src: size=(1, 512, T)
        """
        input = torch.einsum('bdt->tbd', src)
        # input = self.pos_encoder(input)  # (64, 1, 512)
        # input = self.layer_norm(input)
        output = self.transformer_encoder(input) # (64, 1, 512)
        output = self.decoder(output)   # (64, 1, 15)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output