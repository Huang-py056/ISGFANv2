# Model/featureextractor.py
import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F


class GlobalLocalFusionAttention1D(nn.Module):
    def __init__(self, dim, heads=4, attn_dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(dim * 3, dim),
            nn.Sigmoid(),
        )
        self.local_dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.local_pwconv = nn.Conv1d(dim, dim, kernel_size=1)
        self.local_bn = nn.BatchNorm1d(dim)
        self.local_gate = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Linear(dim, dim)
        self.layer_scale = nn.Parameter(torch.ones(dim) * 1e-3)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        h = self.norm(x)
        x_t = h.transpose(1, 2)
        global_token = self.global_pool(x_t).squeeze(-1)
        global_gate = self.global_mlp(global_token).unsqueeze(1)

        local_feat = self.local_dwconv(x_t)
        local_feat = self.local_pwconv(local_feat)
        local_feat = self.local_bn(local_feat)
        local_feat = F.gelu(local_feat)
        local_gate = torch.sigmoid(self.local_gate(local_feat)).transpose(1, 2)

        fused = h * global_gate * local_gate
        fused = self.dropout(self.out_proj(fused)) * self.layer_scale.unsqueeze(0).unsqueeze(0)
        return x + fused


class TransformerFFNAttention1D(nn.Module):
    def __init__(self, dim, heads=4, attn_dropout=0.1, mlp_ratio=4, pooled_length=16):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.dwconv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dwconv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dwconv7 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.mix_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 4, 8), dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(attn_dropout)
        self.layer_scale1 = nn.Parameter(torch.ones(dim) * 1e-4)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(attn_dropout),
        )
        self.layer_scale2 = nn.Parameter(torch.ones(dim) * 1e-4)

    def forward(self, x):
        h = self.norm1(x).transpose(1, 2)
        mixed = (self.dwconv3(h) + self.dwconv5(h) + self.dwconv7(h)) / 3.0
        mixed = self.mix_proj(F.gelu(mixed))
        channel_token = mixed.mean(dim=-1)
        channel_gate = self.channel_gate(channel_token).unsqueeze(-1)
        attn_like = (mixed * channel_gate).transpose(1, 2)
        x = x + self.dropout1(self.out_proj(attn_like)) * self.layer_scale1.unsqueeze(0).unsqueeze(0)
        x = x + self.ffn(self.norm2(x)) * self.layer_scale2.unsqueeze(0).unsqueeze(0)
        return x


class TransformerFFNAttention1DLegacy(nn.Module):
    def __init__(self, dim, heads=4, attn_dropout=0.1, mlp_ratio=4):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(attn_dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


def build_attention_block(mode, dim, heads=4, attn_dropout=0.1):
    if mode == 'attn1':
        return GlobalLocalFusionAttention1D(dim, heads=heads, attn_dropout=attn_dropout)
    if mode == 'attn2':
        return TransformerFFNAttention1D(dim, heads=heads, attn_dropout=attn_dropout)
    if mode == 'attn2_mha':
        return TransformerFFNAttention1DLegacy(dim, heads=heads, attn_dropout=attn_dropout)
    return None

class FIFEBlock(nn.Module):

    def __init__(self, dim, drop_path=0. , dropout_prob=0.1, attention_mode='none', attn_heads=4, attn_dropout=0.1):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.attention_mode = attention_mode
        self.attn_block = build_attention_block(attention_mode, dim, heads=attn_heads, attn_dropout=attn_dropout)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        if self.attn_block is not None:
            x = self.attn_block(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        #x = self.dropout(x)

        return input + self.drop_path(x)

class FRFEBlock(nn.Module):

    def __init__(self, dim, drop_path=0., dropout_prob=0.2, attention_mode='none', attn_heads=4, attn_dropout=0.1):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.attention_mode = attention_mode
        self.attn_block = build_attention_block(attention_mode, dim, heads=attn_heads, attn_dropout=attn_dropout)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        if self.attn_block is not None:
            x = self.attn_block(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        #x = self.dropout(x)

        return input + self.drop_path(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class GRN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class FiFE(nn.Module):
    def __init__(self, in_chans=1, depths=[1, 1, 1, 1], dims=[40, 80, 160, 320], drop_path_rate=0.,
                 attention_mode='none', attn_heads=4, attn_dropout=0.1):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # 迭代构建 3 个降采样层
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[FIFEBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    attention_mode=attention_mode,
                    attn_heads=attn_heads,
                    attn_dropout=attn_dropout,
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        return self.forward_features(x)

class FrFE(nn.Module):
    def __init__(self, in_chans=1, depths=[1, 1, 3, 1], dims=[40, 80, 160, 320], drop_path_rate=0.,
                 attention_mode='none', attn_heads=4, attn_dropout=0.1):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),  # 保持原下采样
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        #self.cab_layers = nn.ModuleList([CAB(dims[i]) for i in range(4)])

        for i in range(4):
            stage = nn.Sequential(
                *[FRFEBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    attention_mode=attention_mode,
                    attn_heads=attn_heads,
                    attn_dropout=attn_dropout,
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            #x = self.cab_layers[i](x)
        return x

    def forward(self, x):
        return self.forward_features(x)  # 仅返回特征

def FRFE(**kwargs):
    return FrFE(depths=[1, 1, 3, 1], dims=[40, 80, 160, 320], **kwargs)

def FIFE(**kwargs):
    return FiFE(depths=[1, 1, 1, 1], dims=[40, 80, 160, 320], **kwargs)

def test_models():  # useless, only for debug

    batch_size = 32
    in_channels = 1
    signal_length = 2048

    original_model = FIFE()
    bearing_model = FRFE()

    orig_params = sum(p.numel() for p in original_model.parameters())
    bear_params = sum(p.numel() for p in bearing_model.parameters())

    print(f"o: {orig_params:,}")
    print(f"s: {bear_params:,}")
    print(f"c: {(bear_params - orig_params) / orig_params * 100:.2f}%")

    # 测试输入输出
    test_input = torch.randn(batch_size, in_channels, signal_length)
    with torch.no_grad():
        orig_output = original_model(test_input)
        bear_output = bearing_model(test_input)

    print(f"i: {test_input.shape}")
    print(f"o: {orig_output.shape}")
    print(f"o: {bear_output.shape}")

    return original_model, bearing_model

if __name__ == "__main__":
    test_models()
