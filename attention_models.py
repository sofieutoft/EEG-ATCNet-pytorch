import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLSA(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.5):
        super(MultiHeadAttentionLSA, self).__init__(embed_dim, num_heads, dropout=dropout)
        self.tau = nn.Parameter(torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32)))

    def forward(self, query, key, value, attn_mask=None):
        query = query * (1.0 / self.tau)
        return super(MultiHeadAttentionLSA, self).forward(query, key, value, attn_mask=attn_mask)

class AttentionModel(nn.Module):
    def __init__(self, key_dim=8, num_heads=2, dropout=0.5, ratio=8):
        super(AttentionModel, self).__init__()
        self.mha_block = MultiHeadAttentionLSA(key_dim, num_heads, dropout=dropout)
        self.ratio = ratio

    def forward(self, input_feature, attention_model):
        if attention_model == 'mha':
            return self.mha_block(input_feature, input_feature, input_feature)
        elif attention_model == 'mhla':
            return self.mha_block(input_feature, input_feature, input_feature)
        elif attention_model == 'se':
            return self.se_block(input_feature)
        elif attention_model == 'cbam':
            return self.cbam_block(input_feature)
        else:
            raise ValueError(f"'{attention_model}' is not a supported attention module!")

    def se_block(self, input_feature):
        channel = input_feature.size(1)
        se_feature = F.adaptive_avg_pool2d(input_feature, (1, 1))
        se_feature = F.relu(se_feature)
        se_feature = F.adaptive_max_pool2d(se_feature, (1, 1))
        se_feature = F.sigmoid(se_feature)
        se_feature = se_feature.expand_as(input_feature)
        return input_feature * se_feature

    def cbam_block(self, input_feature):
        channel_avg = torch.mean(input_feature, dim=(2, 3), keepdim=True)
        channel_max, _ = torch.max(input_feature, dim=(2, 3), keepdim=True)
        channel_wise = torch.cat([channel_avg, channel_max], dim=1)
        channel_wise = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)(channel_wise)
        channel_wise = F.sigmoid(channel_wise)
        return input_feature * channel_wise

# Example usage:
input_feature = torch.randn(1, 3, 64, 8)  # Assuming input size
model = AttentionModel()
#output = model(input_feature, 'mha')  # Change the attention model as needed
output = model(input_feature[:, :, 0], 'mha')