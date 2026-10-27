"""Kernel backbones and the CDG-MIL aggregation head."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_BANDS, FEAT_DIM


class BackboneCNN1D(nn.Module):
    """1-D CNN. (N, N_BANDS) -> (N, FEAT_DIM)"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,   32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32,  64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(128, FEAT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(x.unsqueeze(1)))


class BackboneTransformer(nn.Module):
    """Spectral Transformer. (N, N_BANDS) -> (N, FEAT_DIM)"""
    def __init__(self, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = nn.Parameter(torch.zeros(1, N_BANDS, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.out_proj    = nn.Linear(d_model, FEAT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.input_proj(x.unsqueeze(-1)) + self.pos_enc
        return self.out_proj(self.transformer(t).mean(dim=1))


class BackboneCNN3D(nn.Module):
    """3-D CNN. (N, 50, 50, N_BANDS) -> (N, FEAT_DIM)"""
    def __init__(self, inst_batch: int = 4):
        super().__init__()
        self.inst_batch = inst_batch
        self.cnn = nn.Sequential(
            nn.Conv3d(1,  8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.GroupNorm(1, 8), nn.GELU(), nn.MaxPool3d(2, 2),
            nn.Conv3d(8,  16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.GroupNorm(2, 16), nn.GELU(), nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 32), nn.GELU(), nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(32, FEAT_DIM), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).unsqueeze(1)
        feats = [self.cnn(x[i:i+self.inst_batch]) for i in range(0, x.size(0), self.inst_batch)]
        return self.proj(torch.cat(feats, dim=0))


class BackboneDCNN(nn.Module):
    """Dilated 1-D CNN. (N, N_BANDS) -> (N, FEAT_DIM)"""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=1), nn.GroupNorm(1, channels), nn.GELU(),
        )
        self.blocks = nn.ModuleList([self._block(channels, d) for d in [1, 2, 4, 8]])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(channels, FEAT_DIM), nn.GELU())

    @staticmethod
    def _block(ch, dilation):
        return nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=dilation*2, dilation=dilation, groups=ch),
            nn.Conv1d(ch, ch, 1), nn.GroupNorm(1, ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.input_proj(x.unsqueeze(1))
        for blk in self.blocks:
            t = t + blk(t)[..., :t.size(-1)]
        return self.proj(self.pool(t))


class _SelectiveScan(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner; self.d_state = d_state
        A_init = torch.arange(1, d_state+1, dtype=torch.float).repeat(d_inner, 1)
        self.A_log   = nn.Parameter(A_init.log())
        self.D       = nn.Parameter(torch.ones(d_inner))
        self.x_proj  = nn.Linear(d_inner, d_state*2+1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, _ = x.shape
        A = -self.A_log.exp()
        dbc = self.x_proj(x)
        dt  = F.softplus(self.dt_proj(dbc[..., :1]))
        B   = dbc[..., 1:1+self.d_state]
        C   = dbc[..., 1+self.d_state:]
        dA  = torch.exp(dt.unsqueeze(-1) * A)
        dB  = dt.unsqueeze(-1) * B.unsqueeze(2)
        h   = torch.zeros(N, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys  = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)
            ys.append((h * C[:, i].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1) + x * self.D


class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = int(d_model * expand); self.d_inner = d_inner
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, d_inner*2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.ssm      = _SelectiveScan(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x; x = self.norm(x)
        xz = self.in_proj(x); x_, z = xz.split(self.d_inner, dim=-1)
        x_ = self.conv1d(x_.transpose(1,2)).transpose(1,2)[:, :r.size(1)]
        return self.out_proj(self.ssm(F.silu(x_)) * F.silu(z)) + r


class BackboneMamba(nn.Module):
    """Mamba (selective state space). (N, N_BANDS) -> (N, FEAT_DIM)"""
    def __init__(self, d_model: int = 32, d_state: int = 16, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.layers     = nn.ModuleList([_MambaBlock(d_model, d_state) for _ in range(num_layers)])
        self.norm       = nn.LayerNorm(d_model)
        self.out_proj   = nn.Linear(d_model, FEAT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.input_proj(x.unsqueeze(-1))
        for layer in self.layers: t = layer(t)
        return self.out_proj(self.norm(t).mean(dim=1))


class _KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim
        self.grid_size = grid_size; self.spline_order = spline_order
        n_basis = grid_size + spline_order
        self.register_buffer('grid', torch.linspace(-1.01, 1.01, grid_size + 2*spline_order + 1))
        self.coeff       = nn.Parameter(torch.randn(out_dim, in_dim, n_basis) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) / math.sqrt(in_dim))

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        g = self.grid
        for _ in range(len(x.shape) - 1): g = g.unsqueeze(0)
        b = ((x >= g[..., :-1]) & (x < g[..., 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            dl = g[..., k:-1]   - g[..., :-(k+1)] + 1e-8
            dr = g[..., k+1:]   - g[..., 1:-k]   + 1e-8
            b  = (x - g[..., :-(k+1)]) / dl * b[..., :-1] + (g[..., k+1:] - x) / dr * b[..., 1:]
        return b[..., :self.grid_size + self.spline_order]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(x), self.base_weight) + torch.einsum('...ig,oig->...o', self._basis(torch.tanh(x)), self.coeff)


class BackboneKAN(nn.Module):
    """KAN. (N, N_BANDS) -> (N, FEAT_DIM)"""
    def __init__(self, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.layer1 = _KANLayer(N_BANDS, 64, grid_size, spline_order)
        self.norm1  = nn.LayerNorm(64);   self.drop1 = nn.Dropout(0.3)
        self.layer2 = _KANLayer(64, FEAT_DIM, grid_size, spline_order)
        self.norm2  = nn.LayerNorm(FEAT_DIM); self.drop2 = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.gelu(self.norm1(self.layer1(x))))
        return self.drop2(F.gelu(self.norm2(self.layer2(x))))


class CDGHead(nn.Module):
    """Contextual Deviation-Guided MIL head.

    1. Gated attention -> bag context c
    2. delta_i = h_i - c;  h_i' = h_i + sigmoid(W_g [h_i; delta_i]) * tanh(W_r delta_i)
    3. Second gated attention on h_i' -> regression head
    """
    def __init__(self, att_dim: int = 64):
        super().__init__()
        self.V1 = nn.Linear(FEAT_DIM, att_dim, bias=False)
        self.U1 = nn.Linear(FEAT_DIM, att_dim, bias=False)
        self.w1 = nn.Linear(att_dim, 1, bias=False)

        self.W_g = nn.Linear(2 * FEAT_DIM, FEAT_DIM, bias=True)
        self.W_r = nn.Linear(FEAT_DIM, FEAT_DIM, bias=False)

        self.V2 = nn.Linear(FEAT_DIM, att_dim, bias=False)
        self.U2 = nn.Linear(FEAT_DIM, att_dim, bias=False)
        self.w2 = nn.Linear(att_dim, 1, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(FEAT_DIM, 32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )

    def _attn(self, V, U, w, h):
        gated = torch.tanh(V(h)) * torch.sigmoid(U(h))
        attn  = F.softmax(w(gated), dim=0)
        return (attn * h).sum(dim=0), attn.squeeze(1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        c, _  = self._attn(self.V1, self.U1, self.w1, feat)

        delta  = feat - c.unsqueeze(0)
        gate   = torch.sigmoid(self.W_g(torch.cat([feat, delta], dim=-1)))
        h_ref  = feat + gate * torch.tanh(self.W_r(delta))

        z, _  = self._attn(self.V2, self.U2, self.w2, h_ref)
        return self.fc(z).squeeze()

    def get_attention_weights(self, feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            c, _  = self._attn(self.V1, self.U1, self.w1, feat)
            delta  = feat - c.unsqueeze(0)
            gate   = torch.sigmoid(self.W_g(torch.cat([feat, delta], dim=-1)))
            h_ref  = feat + gate * torch.tanh(self.W_r(delta))
            _, attn = self._attn(self.V2, self.U2, self.w2, h_ref)
        return attn

    def get_gate_map(self, feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            c, _  = self._attn(self.V1, self.U1, self.w1, feat)
            delta  = feat - c.unsqueeze(0)
            return torch.sigmoid(self.W_g(torch.cat([feat, delta], dim=-1)))


class DLModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.head.get_attention_weights(self.backbone(x))

    def get_gate_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.head.get_gate_map(self.backbone(x))


_BACKBONE_REGISTRY = {
    'CNN1D':       BackboneCNN1D,
    'CNN3D':       BackboneCNN3D,
    'Transformer': BackboneTransformer,
    'DCNN':        BackboneDCNN,
    'Mamba':       BackboneMamba,
    'KAN':         BackboneKAN,
}


def get_head() -> nn.Module:
    return CDGHead()


def build_model(backbone_name: str, head: nn.Module) -> DLModel:
    if backbone_name not in _BACKBONE_REGISTRY:
        raise ValueError(f'Unknown backbone: {backbone_name}. Choose from {list(_BACKBONE_REGISTRY)}')
    return DLModel(_BACKBONE_REGISTRY[backbone_name](), head)
