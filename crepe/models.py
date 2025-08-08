from dataclasses import dataclass
from typing import Optional, Tuple, Union, List # Added List
import pandas as pd
import numpy as np
import inspect

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CategoryErrorRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    all_logits: Optional[Tuple[torch.FloatTensor, ...]] = None # Tuple to hold 6 logit tensors (n_a to n_f)
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertForCategoryErrorRegression(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # self.config = config
        self.bert = AutoModel.from_pretrained(config._name_or_path, config=config)
        classifier_dropout = (
            getattr(config, "classifier_dropout", None)         # BERT >= v4
            or getattr(config, "hidden_dropout_prob", None)     # classic BERT
            or getattr(config, "dropout", 0.1)                  # DistilBERT
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.num_error_categories = config.num_error_categories
        self.use_poisson_loss = config.use_poisson_loss
        self.use_presence_loss = config.use_presence_loss
        self.presence_loss_weight = config.presence_loss_weight
        
        self.error_regressors = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for _ in range(self.num_error_categories)]
        )
        self.error_presence = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for _ in range(self.num_error_categories)]
        )

        self._init_regressor_weights()

    def _init_regressor_weights(self):
        def init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        for presence in self.error_presence:
            init_weights(presence)
        for regressor in self.error_regressors:
            init_weights(regressor)

    # def forward(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     token_type_ids: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    #     head_mask: Optional[torch.Tensor] = None,
    #     inputs_embeds: Optional[torch.Tensor] = None,
    #     labels: Optional[torch.Tensor] = None, # Expect labels shape (batch_size, 6) -> n_a to n_f
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple[torch.Tensor], CategoryErrorRegressionOutput]:
    def forward(self, **kwargs):
        input_ids          = kwargs.get("input_ids")
        attention_mask     = kwargs.get("attention_mask")
        token_type_ids     = kwargs.get("token_type_ids", None)
        position_ids       = kwargs.get("position_ids", None)
        head_mask          = kwargs.get("head_mask")
        inputs_embeds      = kwargs.get("inputs_embeds")
        labels             = kwargs.get("labels")
        output_attentions  = kwargs.get("output_attentions")
        output_hidden_states = kwargs.get("output_hidden_states")
        return_dict        = kwargs.get("return_dict",
                                        self.config.use_return_dict)

        bert_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sig = inspect.signature(self.bert.forward).parameters
        if "token_type_ids" in sig and token_type_ids is not None:
            bert_kwargs["token_type_ids"] = token_type_ids 
        if "position_ids" in sig and position_ids is not None:
            bert_kwargs["position_ids"] = position_ids

        outputs = self.bert(**bert_kwargs)
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        pooled_output = (
            outputs.pooler_output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0]
        )

        pooled_output = self.dropout(pooled_output)

        #--
        count_logits = tuple(reg(pooled_output) for reg in self.error_regressors)
        presence_logits = tuple(head(pooled_output) for head in self.error_presence) if self.use_presence_loss else None
        
        if self.use_poisson_loss:
            count_preds = tuple(F.softplus(l) for l in count_logits) # λ ≥ 0
        else:
            count_preds = count_logits

        loss = None
        if labels is not None:
            labels = labels.to(count_preds[0].dtype)

            # -------- ① count 손실 --------
            reg_loss_fct = nn.MSELoss()
            if self.use_poisson_loss:
                reg_loss_fct = nn.PoissonNLLLoss(log_input=False, reduction='mean')
            reg_total = sum(
                reg_loss_fct(count_preds[i].squeeze(-1), labels[:, i])
                for i in range(self.num_error_categories)
            )
            reg_loss = reg_total / self.num_error_categories

            # -------- ② presence 손실 (선택) --------
            if self.use_presence_loss:
                presence_labels = (labels > 0).float()
                bce_fct = nn.BCEWithLogitsLoss()
                bce_total = sum(
                    bce_fct(presence_logits[i].squeeze(-1), presence_labels[:, i])
                    for i in range(self.num_error_categories)
                )
                bce_loss = bce_total / self.num_error_categories
                loss = ((1 - self.presence_loss_weight) * reg_loss +
                        self.presence_loss_weight * bce_loss)
            else:
                loss = reg_loss
        #--

        if not return_dict:
            output = count_preds + (outputs.hidden_states if output_hidden_states else ()) +\
                     (outputs.attentions if output_attentions else ())
            return ((loss,) + output) if loss is not None else output

        return CategoryErrorRegressionOutput(
            loss=loss,
            all_logits=count_preds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def get_model_and_tokenizer(ckpt_path, device=None):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    config = AutoConfig.from_pretrained(ckpt_path)
    model = BertForCategoryErrorRegression.from_pretrained(ckpt_path, config=config)
    if device is not None:
        model.to(device)
    model.eval()
    return model, tokenizer

def get_predicted_counts(model, tokenizer, gt, pred, device=None):
    inputs = tokenizer(
        gt,
        pred,
        return_tensors="pt",
        padding='max_length', # Pad to max_length
        truncation=True,
        max_length=512
    )
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True) 
        if outputs.all_logits is None or len(outputs.all_logits) != model.num_error_categories:
            raise ValueError("Model did not return the expected number of logits.")
        predicted_counts_raw = [logit.item() for logit in outputs.all_logits]
        # Clip predictions to be non-negative
        predicted_counts = [max(0, count) for count in predicted_counts_raw]
        return predicted_counts