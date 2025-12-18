from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MetaCLIPClassifierConfig:
    model_name: str = "facebook/metaclip-b16-fullcc2.5b"
    num_classes: int = 3
    freeze_vision: bool = True
    dropout: float = 0.2
    hidden_dim: int = 256
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


class MetaCLIPImageClassifier(nn.Module):
    """
    MetaCLIP image classifier:
    - HuggingFace AutoModel (CLIP-like dual encoder) -> vision_model
    - Take CLS / pooled vision feature
    - Small MLP head -> class logits
    """

    def __init__(self, cfg: MetaCLIPClassifierConfig):
        super().__init__()
        self.cfg = cfg

        try:
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise ImportError("transformers is required. Install with: pip install transformers") from e

        clip = AutoModel.from_pretrained(cfg.model_name)
        if not hasattr(clip, "vision_model"):
            raise ValueError(
                f"Loaded model '{cfg.model_name}' does not expose `.vision_model`. "
                "Expected a CLIP-like dual-encoder model."
            )
        self.clip = clip
        self.vision = clip.vision_model
        self.patch_size = int(getattr(self.vision.config, "patch_size", 14))

        # Optional LoRA fine-tuning (train adapters + head).
        if bool(cfg.use_lora):
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except Exception as e:  # pragma: no cover
                raise ImportError("peft is required for LoRA. Install with: pip install peft") from e

            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(cfg.lora_r),
                lora_alpha=int(cfg.lora_alpha),
                lora_dropout=float(cfg.lora_dropout),
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self.clip = get_peft_model(self.clip, lora_cfg)
            # Refresh vision handle from the wrapped model
            self.vision = self.clip.get_submodule("vision_model")

            # Training policy:
            # - If freeze_vision=True: train LoRA adapters only (plus head).
            # - If freeze_vision=False: train full vision backbone + LoRA adapters (plus head).
            if cfg.freeze_vision:
                for name, p in self.vision.named_parameters():
                    if "lora_" in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                for p in self.vision.parameters():
                    p.requires_grad = True
        elif cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        # We don't use text encoder for classification; keep it frozen to save memory/compute.
        if hasattr(self.clip, "text_model"):
            for p in self.clip.text_model.parameters():
                p.requires_grad = False

        vision_hidden = int(self.vision.config.hidden_size)
        hidden_dim = int(cfg.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(vision_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(hidden_dim, int(cfg.num_classes)),
        )

        # Head is always trainable
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B,3,H,W) float tensor already normalized for the backbone.
        Returns:
            logits: (B,num_classes)
        """
        # ViT patch embedding expects H/W divisible by patch_size. If not, resize to nearest lower divisible.
        if pixel_values.dim() == 4:
            h, w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
            ps = int(self.patch_size)
            if ps > 0 and ((h % ps) != 0 or (w % ps) != 0):
                new_h = max(ps, (h // ps) * ps)
                new_w = max(ps, (w // ps) * ps)
                pixel_values = F.interpolate(pixel_values, size=(new_h, new_w), mode="bilinear", align_corners=False)

        out = self.vision(pixel_values=pixel_values, interpolate_pos_encoding=True)
        # Prefer pooler_output if present; else use CLS token.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0, :]
        feat = F.normalize(feat, dim=-1)
        return self.head(feat)


