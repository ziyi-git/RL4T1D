# agents/models/timerxl_openltm_glucose.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# open-ltm 的 Timer-XL
from third_party.openltm.models.timer_xl import Model as TimerXLModel


class _TimerXLConfig:
    """最小配置对象，仿 open-ltm 的 configs，Model 只会读到这些字段。"""
    def __init__(self, **kwargs):
        # 架构/功能
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
        # 预训练/精调
        self.pretrained_path        = kwargs.get('pretrained_path', '')
        self.pretrained_strict      = kwargs.get('pretrained_strict', False)
        self.map_location           = kwargs.get('map_location', 'cpu')
        self.freeze_backbone        = kwargs.get('freeze_backbone', True)
        self.finetune_head          = kwargs.get('finetune_head', True)
        self.dtype                  = kwargs.get('dtype', 'fp32')   # 'fp32' | 'bf16' | 'fp16'


class TimerXLGlucoseHead(nn.Module):
    """
    Tap-out 适配器（μ 分支）：
    - 输入: concat_state_action = [extract_state; action]  -> in_proj -> x_seq[B,1,1]
    - Timer-XL 输出 dec_out[B,1,1]，经 tanh → mu ∈ [-1, 1]
    - 预训练权重：可选加载，自定义冻结策略
    """
    def __init__(self, d_in_scalar: int, timer_cfg: _TimerXLConfig):
        super().__init__()
        self.cfg = timer_cfg

        # 1) Timer-XL 主体
        self.timer = TimerXLModel(timer_cfg)

        # 2) in_proj：把 [d_h + d_a] 压到单标量，作为单变量单 token 的输入
        # self.in_proj = nn.Linear(d_in_scalar, 1)
        self.patch_len = int(self.cfg.input_token_len)  # 96
        self.in_proj = nn.Linear(d_in_scalar, self.patch_len)  # [B, d_in] -> [B, L=patch_len]

        # 3) 可选：加载预训练权重
        if self.cfg.pretrained_path:
            self._load_pretrained(self.cfg.pretrained_path,
                                  strict=self.cfg.pretrained_strict,
                                  map_location=self.cfg.map_location)

        # 4) dtype 统一（避免混精度不一致）
        if self.cfg.dtype.lower() == 'bf16':
            self.timer = self.timer.to(dtype=torch.bfloat16)
        elif self.cfg.dtype.lower() == 'fp16':
            self.timer = self.timer.to(dtype=torch.float16)
        else:
            self.timer = self.timer.to(dtype=torch.float32)

        # 5) 冻结策略
        if self.cfg.freeze_backbone:
            for n, p in self.timer.named_parameters():
                # 如果你希望 head 仍可调，且 finetune_head=True，则跳过 head
                if (not self.cfg.finetune_head) or (not n.startswith('head.')):
                    p.requires_grad = False

    @torch.no_grad()
    def _sanitize_state_dict_keys(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """去掉常见前缀，尽量对齐 open-ltm 的 Timer-XL 模块命名。"""
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for pref in ('module.', 'model.', 'timer.', 'backbone.'):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            new_sd[nk] = v
        return new_sd

    def _load_pretrained(self, path: str, strict: bool, map_location: str):
        if not os.path.isfile(path):
            print(f"[TimerXL] WARNING: pretrained checkpoint not found: {path}")
            return
        print(f"[TimerXL] Loading pretrained checkpoint: {path} (strict={strict})")
        ckpt = torch.load(path, map_location=map_location)
        # 兼容不同保存格式
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                sd = ckpt['state_dict']
            elif 'model' in ckpt and isinstance(ckpt['model'], dict):
                sd = ckpt['model']
            else:
                # 直接就是 state_dict
                sd = ckpt
        else:
            sd = ckpt
        
        own = self.timer.state_dict()
        sd = self._sanitize_state_dict_keys(sd)
        # 只保留形状完全一致的键
        filtered = {k: v for k, v in sd.items() if (k in own) and (own[k].shape == v.shape)}
        # 只把能对上的键加载（head 维度不同时会被忽略）
        # missing, unexpected = self.timer.load_state_dict(sd, strict=False)
        missing, unexpected = self.timer.load_state_dict(filtered, strict=False)
        # print(f"[TimerXL] loaded with strict=False. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        print(f"[TimerXL] loaded (filtered). matched={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if strict and (missing or unexpected):
            # 如果用户强制 strict，则抛错
            raise RuntimeError(f"[TimerXL] strict load failed. missing={missing[:5]}..., unexpected={unexpected[:5]}...")

    def forward(self, concat_state_action: torch.Tensor) -> torch.Tensor:
        """
        输入: concat_state_action [B, d_h + d_a]
        输出: mu [B, 1]  (范围 [-1,1])
        """
        B = concat_state_action.size(0)
        # x_scalar = self.in_proj(concat_state_action)         # [B, 1]
        # x_seq = x_scalar.view(B, 1, 1)                       # [B, L=1, C=1]
        patch = self.in_proj(concat_state_action)
        x_seq = patch.view(B, self.patch_len, 1)
        # dec_out = self.timer(x_seq, None, None)              # [B, 1, 1] (fp32/bf16/fp16 由 cfg.dtype 控制)
        dec_out = self.timer(x_seq, None, None)             # [B, L_out=96, C=1]  (因为 output_token_len=96)
        print(f"TimerXLGlucoseHead forward: dec_out shape: {dec_out.shape}, dtype: {dec_out.dtype}")
        # 如果 timer 是混精度，tanh 在 fp32 计算更稳，可视需要转回
        # mu = torch.tanh(dec_out[:, -1:, 0:1].to(dtype=torch.float32))  # [B, 1]
        # mu = torch.tanh(dec_out[:, -1:, 0:1].to(torch.float32))  # 取最后一个时间步→ [B,1] torch.Size([1, 1, 1])
        mu = torch.tanh(dec_out[:, -1:, 0].to(torch.float32))  # 取最后一个时间步→ [B,1] torch.Size([1, 1])
        print(f"TimerXLGlucoseHead forward: mu shape: {mu.shape}, dtype: {mu.dtype}")
        return mu