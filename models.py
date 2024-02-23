import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(0, 0))
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 22), padding=(0, 0))
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        self.pooling = nn.AvgPool2d(kernel_size=(pool_size, 1), stride=(pool_size, 1))
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, attention_type='mha'):
        super(AttentionBlock, self).__init__()
        self.attention_type = attention_type
        if attention_type == 'mha':
            self.attention = nn.MultiheadAttention(in_channels, 1)
        elif attention_type == 'se':
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.attention_type == 'mha':
            x = x.permute(2, 0, 1, 3)  # Change to (seq_len, batch, channels, 1)
            x, _ = self.attention(x, x, x)
            x = x.permute(1, 2, 0, 3)  # Change back to (batch, channels, seq_len, 1)
        elif self.attention_type == 'se':
            weights = self.attention(x)
            x = x * weights
        return x

class TemporalConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.batch_norm1 = nn.BatchNorm1d(output_channels)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.batch_norm2 = nn.BatchNorm1d(output_channels)

        self.residual_conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.residual_batch_norm = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        residual = self.residual_conv(x)
        residual = self.residual_batch_norm(residual)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)

        x += residual
        x = self.activation(x)

        return x

class ATCNet(nn.Module):
    def __init__(self, n_classes, in_chans=22, in_samples=1125, n_windows=5, attention='mha',
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 tcn_activation='elu', fuse='average'):
        super(ATCNet, self).__init__()

        self.conv_block = ConvBlock(1, eegn_F1, eegn_kernelSize, eegn_poolSize, eegn_dropout)

        self.temporal_blocks = nn.ModuleList([
            TemporalConvBlock(eegn_F1 * eegn_D, tcn_filters, tcn_kernelSize, tcn_dropout)
            for _ in range(n_windows)
        ])

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(tcn_filters, attention_type=attention)
            for _ in range(n_windows)
        ])

        self.fuse = fuse

        if fuse == 'average':
            self.final_conv = nn.Conv1d(tcn_filters * n_windows, n_classes, kernel_size=1)
        elif fuse == 'concatenate':
            self.final_conv = nn.Conv1d(tcn_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_block(x)

        x_temporal_blocks = []
        for i in range(len(self.temporal_blocks)):
            x_temporal_blocks.append(self.temporal_blocks[i](x))

        x_temporal_blocks = torch.stack(x_temporal_blocks, dim=-1)

        x_attention_blocks = []
        for i in range(len(self.attention_blocks)):
            x_attention_blocks.append(self.attention_blocks[i](x_temporal_blocks[:, :, :, i]))

        x_attention_blocks = torch.stack(x_attention_blocks, dim=-1)

        if self.fuse == 'average':
            x_attention_blocks = x_attention_blocks.mean(dim=-1)
            x = self.final_conv(x_attention_blocks)
        elif self.fuse == 'concatenate':
            x = self.final_conv(x_attention_blocks)

        return x.squeeze()


# Instantiate the model
model = ATCNet(n_classes=4)

# Print the model architecture
print(model)
