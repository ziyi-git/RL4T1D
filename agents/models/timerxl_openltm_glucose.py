# agents/models/timerxl_openltm_glucose.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 来自我们刚 vendor 的 open-ltm 子包
from third_party.openltm.models.timer_xl import Model as TimerXLModel

class _TimerXLConfig:
    """最小配置对象，仿 open-ltm 的 configs，Model 只会读到这些字段。"""
    def __init__(self, **kwargs):
        # 必需字段（见 open-ltm/models/timer_xl.py）
        self.input_token_len   = kwargs.get('input_token_len', 1)
        self.output_token_len  = kwargs.get('output_token_len', 1)
        self.covariate         = kwargs.get('covariate', False)
        self.flash_attention   = kwargs.get('flash_attention', False)
        self.d_model           = kwargs.get('d_model', 384)
        self.n_heads           = kwargs.get('n_heads', 6)
        self.d_ff              = kwargs.get('d_ff', 1024)
        self.e_layers          = kwargs.get('e_layers', 4)
        self.dropout           = kwargs.get('dropout', 0.0)
        self.activation        = kwargs.get('activation', 'gelu')
        self.output_attention  = kwargs.get('output_attention', False)
        self.use_norm          = kwargs.get('use_norm', False)

class TimerXLGlucoseHead(nn.Module):
    """
    Tap-out 适配器：把 [extract_state; action] 向量压成单标量序列，喂给 open-ltm 的 Timer-XL。
    - mu 由 Timer-XL 预测（tanh 到 [-1, 1]）
    - sigma 仍沿用 RL4T1D 原始的 MLP 头（在 GlucoseModel 中计算）
    """
    def __init__(self, d_in_scalar, timer_cfg: _TimerXLConfig):
        super().__init__()
        self.timer = TimerXLModel(timer_cfg)
        # 把 [extract_state; action] 压到单标量（作为单变量单 token 的取值）
        self.in_proj = nn.Linear(d_in_scalar, 1)

    def forward(self, concat_state_action: torch.Tensor) -> torch.Tensor:
        """
        输入: concat_state_action [B, d_h + d_a]
        输出: mu [B, 1]  (范围 [-1,1])
        """
        B = concat_state_action.size(0)
        x_scalar = self.in_proj(concat_state_action)        # [B, 1]
        x_seq = x_scalar.view(B, 1, 1)                      # [B, L=1, C=1]
        dec_out = self.timer(x_seq, None, None)             # [B, 1, 1]
        mu = torch.tanh(dec_out[:, -1:, 0:1])               # [B, 1]
        return mu