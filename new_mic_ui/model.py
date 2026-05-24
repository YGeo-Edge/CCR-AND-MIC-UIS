"""
InternVLClassifier: vision encoder (InternViT-300M) + LoRA + linear classification head.
Only the vision encoder is loaded; the language model is discarded to save VRAM.
"""
import json
import os

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Transformers 5.x added all_tied_weights_keys but the InternVL custom model code
# (trust_remote_code) was written for 4.x and never sets it on the top-level class.
# Patch _move_missing_keys_from_meta_to_device to tolerate the missing attribute.
try:
    import transformers.modeling_utils as _mu
    _orig_move = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device

    def _compat_move(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_move(self, *args, **kwargs)

    _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _compat_move
except AttributeError:
    pass  # transformers < 5.x, not needed



class InternVLClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_id = model_id
        self.num_classes = num_classes

        if lora_target_modules is None:
            lora_target_modules = ["qkv", "proj"]

        print(f"Loading {model_id} ...")
        full_model = AutoModel.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # avoids flash_attn dependency
        )

        # Extract vision encoder; discard language model to free VRAM
        if hasattr(full_model, "vision_model"):
            self.vision_model = full_model.vision_model
        elif hasattr(full_model, "visual_encoder"):
            self.vision_model = full_model.visual_encoder
        else:
            raise AttributeError(f"Cannot locate vision encoder in {type(full_model).__name__}")
        del full_model
        torch.cuda.empty_cache()

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        hidden_size = cfg.vision_config.hidden_size  # 1024 for InternViT-300M
        print(f"Vision encoder hidden_size={hidden_size}")

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.vision_model = get_peft_model(self.vision_model, lora_cfg)
        self.vision_model.print_trainable_parameters()

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        # CLS token sits at position 0 in InternViT
        features = outputs.last_hidden_state[:, 0, :].float()
        logits = self.classifier(self.dropout(features))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    # ── persistence ──────────────────────────────────────────────────────────

    def save_adapter(self, save_path: str, class_names: list):
        os.makedirs(save_path, exist_ok=True)
        self.vision_model.save_pretrained(save_path)
        torch.save(self.classifier.state_dict(), os.path.join(save_path, "classifier_head.pt"))
        meta = {
            "model_id": self.model_id,
            "num_classes": self.num_classes,
            "class_names": class_names,
        }
        with open(os.path.join(save_path, "model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Adapter saved → {save_path}")

    @classmethod
    def load_adapter(cls, adapter_path: str, **kwargs) -> "InternVLClassifier":
        from peft import PeftModel

        with open(os.path.join(adapter_path, "model_meta.json")) as f:
            meta = json.load(f)

        instance = cls(
            model_id=meta["model_id"],
            num_classes=meta["num_classes"],
            **kwargs,
        )
        # Replace LoRA model with saved adapter weights
        base = instance.vision_model.base_model.model
        instance.vision_model = PeftModel.from_pretrained(base, adapter_path)
        instance.classifier.load_state_dict(
            torch.load(os.path.join(adapter_path, "classifier_head.pt"), map_location="cpu")
        )
        return instance
