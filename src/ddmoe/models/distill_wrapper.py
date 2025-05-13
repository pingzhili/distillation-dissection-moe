from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

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
            proxy_model: Union[PreTrainedModel, List[PreTrainedModel]],
            anti_kd_coef: float,
            kd_temperature: float,
            lm_head_projector: bool = False,
            lm_head_projector_dim: int = 512,
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.train()
        self.proxy_model_list = proxy_model if isinstance(proxy_model, list) else [proxy_model]
        self.proxy_model_list = nn.ModuleList(self.proxy_model_list)
        self.vocab_size = teacher_model.config.vocab_size
        self.anti_kd_coef = anti_kd_coef
        self.kd_temperature = kd_temperature

        logger.info(f"Teacher model: {teacher_model.__class__.__name__}")
        logger.info(f"Proxy model: {proxy_model.__class__.__name__}")
        logger.info(f"Anti KD coefficient: {anti_kd_coef}")
        logger.info(f"KD temperature: {kd_temperature}")

        logger.info("Only train TEACHER model's LM head, freeze all others")
        for model in self.proxy_model_list:
            for param in model.parameters():
                param.requires_grad = False

        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        if lm_head_projector:
            self.teacher_model.lm_head.bia_projector = nn.Sequential(
                nn.Linear(self.teacher_model.lm_head.out_features, lm_head_projector_dim, bias=False),
                nn.ReLU(),
                nn.Linear(lm_head_projector_dim, self.teacher_model.lm_head.out_features, bias=False),
            )
            # init the second linear layer's weight to 0
            self.teacher_model.lm_head.bia_projector[-1].weight.data.zero_()
            logger.info(f"Adding lm_head_projector to TEACHER model")
            
            # Modify the forward function of lm_head
            self.teacher_model.lm_head._original_forward = self.teacher_model.lm_head.forward
            def _forward_hook(self, input: torch.Tensor) -> torch.Tensor:
                ret = self._original_forward(input)
                ret = ret + self.bia_projector(ret)
                return ret
            self.teacher_model.lm_head.forward = _forward_hook.__get__(self.teacher_model.lm_head, type(self.teacher_model.lm_head))
            self.teacher_model.lm_head.bia_projector.requires_grad = True
        else:
            for name, param in self.teacher_model.named_parameters():
                if "lm_head" in name:
                    param.requires_grad = True
                logger.info(f"Setting trainable on {name} (set to trainable? {param.requires_grad})")

    def forward(self, *args, **kwargs) -> AntiDistillCausalLMOutputWithPast:
        assert self.training, "AntiDistillWrapper should only be used in training mode"

        teacher_outputs = self.teacher_model(*args, **kwargs)
        with torch.no_grad():
            proxy_outputs_list = []
            for proxy_model in self.proxy_model_list:
                proxy_outputs_list.append(proxy_model(*args, **kwargs))

        lm_loss = teacher_outputs.loss
        teacher_logits = teacher_outputs.logits[..., :self.vocab_size]
        proxy_logits_list = [proxy_outputs.logits[..., :self.vocab_size] for proxy_outputs in proxy_outputs_list]

        teacher_logits = teacher_logits.to(torch.float32)
        proxy_logits_list = [proxy_logits.to(torch.float32) for proxy_logits in proxy_logits_list]

        kd_criteria = nn.KLDivLoss(reduction="batchmean")
        kd_loss_list = [
            kd_criteria(
                F.log_softmax(proxy_logits / self.kd_temperature, dim=-1),
                F.softmax(teacher_logits / self.kd_temperature, dim=-1)
            )
            for proxy_logits in proxy_logits_list
        ]
        kd_loss_list = [kd_loss if not torch.isnan(kd_loss) else 0 for kd_loss in kd_loss_list]
        kd_loss = sum(kd_loss_list) / len(kd_loss_list)
            
        anti_kd_loss = - self.anti_kd_coef * kd_loss

        loss = lm_loss + anti_kd_loss

        return AntiDistillCausalLMOutputWithPast(
            loss=loss, logits=teacher_logits, lm_loss=lm_loss, kd_loss=kd_loss,
            past_key_values=teacher_outputs.past_key_values,
            hidden_states=teacher_outputs.hidden_states,
            attentions=teacher_outputs.attentions
        )
