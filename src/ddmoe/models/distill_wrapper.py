from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class AntiDistillCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    kd_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class AntiDistillWrapper(nn.Module):
    """Model wrapper to train the TEACHER model for anti distillation. """

    def __init__(
            self,
            teacher_model: PreTrainedModel,
            proxy_model: PreTrainedModel,
            anti_kd_coef: float,
            kd_temperature: float
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.train()
        self.proxy_model = proxy_model
        self.vocab_size = min(teacher_model.config.vocab_size, proxy_model.config.vocab_size)
        self.anti_kd_coef = anti_kd_coef
        self.kd_temperature = kd_temperature

        logger.info(f"Teacher model: {teacher_model.__class__.__name__}")
        logger.info(f"Proxy model: {proxy_model.__class__.__name__}")
        logger.info(f"Anti KD coefficient: {anti_kd_coef}")
        logger.info(f"KD temperature: {kd_temperature}")

        logger.info("Only train TEACHER model's LM head, freeze all others")
        for param in self.proxy_model.parameters():
            param.requires_grad = False

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        for name, param in self.teacher_model.named_parameters():
            if "lm_head" in name or "embed_tokens" in name:
                param.requires_grad = True
                logger.info(f"Setting trainable on {name} (set to trainable? {param.requires_grad})")

        total_trainable_params = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_trainable_params}")

    def forward(self, *args, **kwargs) -> AntiDistillCausalLMOutputWithPast:
        assert self.training, "AntiDistillWrapper should only be used in training mode"

        teacher_outputs = self.teacher_model(*args, **kwargs)
        with torch.no_grad():
            proxy_outputs = self.proxy_model(*args, **kwargs)

        lm_loss = teacher_outputs.loss
        teacher_logits = teacher_outputs.logits[..., :self.vocab_size]
        proxy_logits = proxy_outputs.logits[..., :self.vocab_size]

        kd_criteria = nn.KLDivLoss(reduction="batchmean")
        kd_loss = kd_criteria(
            F.log_softmax(proxy_logits / self.kd_temperature, dim=-1),
            F.softmax(teacher_logits / self.kd_temperature, dim=-1)
        )
        anti_kd_loss = - self.anti_kd_coef * kd_loss

        loss = lm_loss + anti_kd_loss

        return AntiDistillCausalLMOutputWithPast(
            loss=loss, logits=teacher_logits, lm_loss=lm_loss, kd_loss=kd_loss,
            past_key_values=teacher_outputs.past_key_values,
            hidden_states=teacher_outputs.hidden_states,
            attentions=teacher_outputs.attentions
        )
