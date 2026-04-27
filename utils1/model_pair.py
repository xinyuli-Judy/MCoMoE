import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,),
                 dilation=(1,), if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=p0,
            dilation=dilation,
            bias=True if if_bias else False
        )
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x


class LearnablePositionalEncoding1D(nn.Module):


    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.position_encoding = nn.Parameter(torch.zeros(1, d_model, max_len))
        nn.init.normal_(self.position_encoding, mean=0.0, std=0.02)

    def forward(self, x):
        # x:[B, C, L]
        seq_len = x.size(2)
        if seq_len > self.position_encoding.size(2):
            pos_enc = self.position_encoding[:, :, :seq_len]
        else:
            pos_enc = F.interpolate(self.position_encoding, size=seq_len, mode='linear', align_corners=False)
        return x + pos_enc



class LocalWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, window_size=16, dropout=0.1,
                 attn_output_gate: str = "headwise"):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert attn_output_gate in ["none", "headwise", "elementwise"]

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.attn_output_gate = attn_output_gate

        if attn_output_gate == "headwise":
            q_out_dim = embed_dim + num_heads
        elif attn_output_gate == "elementwise":
            q_out_dim = embed_dim * 2
        else:
            q_out_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, q_out_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        L, B, C = query.shape
        H, D, W = self.num_heads, self.head_dim, self.window_size

        q_raw = self.q_proj(query.permute(1, 0, 2))
        k = self.k_proj(key.permute(1, 0, 2))
        v = self.v_proj(value.permute(1, 0, 2))

        gate = None
        if self.attn_output_gate == "headwise":
            q, gate_score = q_raw[..., :C], q_raw[..., C:]
            gate = gate_score.view(B, L, H).permute(0, 2, 1).unsqueeze(-1)
        elif self.attn_output_gate == "elementwise":
            q, gate_score = torch.chunk(q_raw, 2, dim=-1)
            gate = gate_score.view(B, L, H, D).permute(0, 2, 1, 3)
        else:
            q = q_raw

        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)

        pad = (W - (L % W)) % W
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Lp = q.size(2)
        nw = Lp // W

        q = q.view(B, H, nw, W, D)
        k = k.view(B, H, nw, W, D)
        v = v.view(B, H, nw, W, D)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).view(B, H, Lp, D)[:, :, :L, :]

        if gate is not None:
            out = out * torch.sigmoid(gate)

        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out).permute(1, 0, 2)

        gate_out = torch.sigmoid(gate) if gate is not None else None
        return out, gate_out


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class FFN1D(nn.Module):
    def __init__(self, channels, expansion=4, dropout=0.1, activation="gelu"):
        super().__init__()
        hidden = channels * expansion

        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttnFFNBlock1D(nn.Module):
    def __init__(self, channels, attn_module, ffn_expansion=4, attn_dropout=0.1, ffn_dropout=0.1, drop_path=0.0):
        super().__init__()
        self.channels = channels
        self.attn = attn_module
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = FFN1D(channels, expansion=ffn_expansion, dropout=ffn_dropout, activation="gelu")
        self.drop_path = DropPath(drop_path)

    def forward(self, x, context):
        x_ln = self.norm_q(x.permute(0, 2, 1)).permute(0, 2, 1)
        ctx_ln = self.norm_kv(context.permute(0, 2, 1)).permute(0, 2, 1)

        q = x_ln.permute(2, 0, 1)
        k = ctx_ln.permute(2, 0, 1)
        v = ctx_ln.permute(2, 0, 1)

        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.permute(1, 2, 0)
        x = x + self.drop_path(attn_out)

        x_ffn_ln = self.norm_ffn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.drop_path(self.ffn(x_ffn_ln))

        return x


class EnhancedSequenceStructureCoAttention(nn.Module):
    def __init__(self, channels, use_local_window=True, window_size=16,
                 ffn_expansion=4, attn_dropout=0.1, ffn_dropout=0.1, drop_path=0.0, use_gate=True):
        super().__init__()
        self.channels = channels
        self.use_gate = use_gate
        self.pos_encoding = LearnablePositionalEncoding1D(channels, max_len=128)


        seq2struct_attn = LocalWindowAttention(embed_dim=channels, num_heads=8, window_size=window_size, dropout=attn_dropout)
        struct2seq_attn = LocalWindowAttention(embed_dim=channels, num_heads=8, window_size=window_size, dropout=attn_dropout)

        self.seq_block = CrossAttnFFNBlock1D(channels=channels, attn_module=seq2struct_attn,
                                             ffn_expansion=ffn_expansion, attn_dropout=attn_dropout,
                                             ffn_dropout=ffn_dropout, drop_path=drop_path)
        self.struct_block = CrossAttnFFNBlock1D(channels=channels, attn_module=struct2seq_attn,
                                                ffn_expansion=ffn_expansion, attn_dropout=attn_dropout,
                                                ffn_dropout=ffn_dropout, drop_path=drop_path)

        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv1d(channels * 2, channels, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, seq_features, struct_features):
        seq = self.pos_encoding(seq_features)
        struct = self.pos_encoding(struct_features)

        seq_enhanced = self.seq_block(seq, struct)
        struct_enhanced = self.struct_block(struct, seq)

        if not self.use_gate:
            return seq_enhanced, struct_enhanced

        combined = torch.cat([seq_enhanced, struct_enhanced], dim=1)
        gate_weights = self.gate(combined)

        seq_output = seq_features + gate_weights * (seq_enhanced - seq_features)
        struct_output = struct_features + (1 - gate_weights) * (struct_enhanced - struct_features)

        return seq_output, struct_output


class LocalWindowConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, window_sizes=[3, 5, 7],
                 same_padding=True, relu=True, bn=True, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_sizes = window_sizes
        self.dropout = dropout

        self.window_conv_branches = nn.ModuleList()
        for win_size in window_sizes:
            padding = (win_size - 1) // 2 if same_padding else 0
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=win_size, stride=1, padding=padding, bias=False),
                nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.window_conv_branches.append(branch)

        self.attention_weights = nn.Parameter(torch.ones(len(window_sizes)))
        self.fusion_conv = nn.Conv1d(out_channels * len(window_sizes), out_channels, kernel_size=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.fusion_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        branch_outputs =[]
        for idx, (win_size, conv_branch) in enumerate(zip(self.window_sizes, self.window_conv_branches)):
            win_feat = conv_branch(x)
            branch_outputs.append(win_feat)

        weights = F.softmax(self.attention_weights, dim=0)
        weighted_outputs =[]
        for idx, feat in enumerate(branch_outputs):
            weight_broadcast = weights[idx].view(1, 1, 1)
            weighted_outputs.append(feat * weight_broadcast)

        fused = torch.cat(weighted_outputs, dim=1)
        output = self.fusion_conv(fused)

        return output


class EnhancedDynamicFeatureFusion(nn.Module):
    def __init__(self, in_channels=256, fusion_dim=256, use_local_window=True):
        super(EnhancedDynamicFeatureFusion, self).__init__()

        self.seq_local_window = LocalWindowConv1D(in_channels, in_channels, window_sizes=[3, 5, 7], same_padding=True)
        self.struct_local_window = LocalWindowConv1D(in_channels, in_channels, window_sizes=[3, 5, 7], same_padding=True)

        self.co_attention = EnhancedSequenceStructureCoAttention(
            channels=in_channels,
            use_local_window=use_local_window,
            window_size=16,
            ffn_expansion=4,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            drop_path=0.1,
            use_gate=True
        )

        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

        self.fusion_weights = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 2),
            nn.Softmax(dim=-1)
        )

        self.fusion_conv = nn.Sequential(
            Conv1d(in_channels * 2, fusion_dim, kernel_size=3, same_padding=True),
            Conv1d(fusion_dim, fusion_dim, kernel_size=3, same_padding=True)
        )
        self.channel_adjust = Conv1d(in_channels, fusion_dim, kernel_size=1, same_padding=True, relu=False)

    def forward(self, seq_features, struct_features, adj):
        seq_enhanced = self.seq_local_window(seq_features)
        struct_enhanced = self.struct_local_window(struct_features)

        struct_gcn = torch.bmm(struct_enhanced, adj)
        struct_enhanced = struct_enhanced + struct_gcn

        seq_coattn, struct_coattn = self.co_attention(seq_enhanced, struct_enhanced)

        seq_pool = torch.mean(seq_coattn, dim=2)
        struct_pool = torch.mean(struct_coattn, dim=2)

        global_context = self.global_proj(seq_pool)
        global_gate = torch.sigmoid(global_context).unsqueeze(-1)

        seq_coattn = seq_coattn * global_gate
        struct_coattn = struct_coattn * global_gate

        weight_input = torch.cat([seq_pool, struct_pool, global_context], dim=1)
        weights = self.fusion_weights(weight_input)

        weighted_fusion = (seq_coattn * weights[:, 0:1].unsqueeze(2) +
                           struct_coattn * weights[:, 1:2].unsqueeze(2))

        weighted_fusion_adjusted = self.channel_adjust(weighted_fusion)

        concatenated = torch.cat([seq_coattn, struct_coattn], dim=1)
        fused = self.fusion_conv(concatenated)

        fused = fused + weighted_fusion_adjusted

        return fused


class MotifAwareAttentionPooling(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_dim // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, C, L]
        attn_logits = self.attention_net(x)  # [B, 1, L]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 1, L]


        pooled_feat = torch.sum(x * attn_weights, dim=-1)  # [B, C]
        return pooled_feat, attn_weights




class MoEClassifier(nn.Module):

    def __init__(self, in_dim, num_classes=1, num_experts=4):
        super().__init__()
        self.num_experts = num_experts


        self.gating_network = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, num_experts),
            nn.Softmax(dim=-1)
        )


        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(in_dim // 2, num_classes)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [B, C]
        gate_weights = self.gating_network(x)  # [B, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  #[B, num_experts, num_classes]


        final_output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return final_output


class MCoMoEModel(nn.Module):
    def __init__(self, kmer_channels=640, structure_channels=1,
                 conv_channels=256, fusion_dim=256, num_classes=1,
                 use_local_window=True):
        super(MCoMoEModel, self).__init__()

        self.conv_seq = Conv1d(kmer_channels, conv_channels, kernel_size=1, same_padding=True)
        self.conv_struct = Conv1d(structure_channels, conv_channels, kernel_size=3, same_padding=True)


        self.dynamic_fusion = EnhancedDynamicFeatureFusion(
            in_channels=conv_channels,
            fusion_dim=fusion_dim,
            use_local_window=use_local_window
        )


        self.motif_pool = MotifAwareAttentionPooling(in_dim=fusion_dim)


        self.moe_classifier = MoEClassifier(in_dim=fusion_dim, num_classes=num_classes, num_experts=4)

    def forward(self, rnafm_3mer, structure, adj):
        if rnafm_3mer.size(2) != structure.size(2):
            rnafm_3mer = F.interpolate(rnafm_3mer, size=structure.size(2), mode='linear', align_corners=False)

        seq_features = self.conv_seq(rnafm_3mer)
        struct_features = self.conv_struct(structure)

        fused_features = self.dynamic_fusion(seq_features, struct_features, adj)


        pooled_feat, attn_weights = self.motif_pool(fused_features)

        logits = self.moe_classifier(pooled_feat)

        return logits, {"motif_attention": attn_weights}


def get_model(args):
    model = MCoMoEModel(
        kmer_channels=getattr(args, 'kmer_channels', 640),
        structure_channels=getattr(args, 'structure_channels', 1),
        conv_channels=getattr(args, 'conv_channels', 256),
        fusion_dim=getattr(args, 'fusion_dim', 256),
        num_classes=getattr(args, 'num_classes', 1),
        use_local_window=getattr(args, 'use_local_window', True)
    )
    return model




