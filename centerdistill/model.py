"""
model.py — CenterDistill model architecture and custom Trainer.

Architecture
------------
A frozen-or-fine-tuned XLM-RoBERTa encoder shared by two heads:
  • span_head   : linear over all token positions → start/end logits
  • center_head : linear on [CLS] token → K-dimensional center logits

Training objective (Equation 4 in the paper)
--------------------------------------------
    L = λ · KL( P_T(c|q) ‖ P_S(c|q,d) )  +  (1−λ) · L_QA(a*, a)

where KL is computed from log-softmax student output and soft teacher labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, TrainingArguments, Trainer
from typing import Dict, Optional


class CenterDistillModel(nn.Module):
    """
    Joint center-prediction + extractive QA model.

    Parameters
    ----------
    base_model_path : HuggingFace model name or local directory.
    num_centers     : K — number of semantic centres (from config).
    lambda_kl       : weight on the KL distillation loss (0 < λ ≤ 1).
    """

    def __init__(
        self,
        base_model_path: str,
        num_centers: int,
        lambda_kl: float = 0.7,
    ):
        super().__init__()
        self.encoder      = AutoModel.from_pretrained(base_model_path)
        hidden            = self.encoder.config.hidden_size

        # Two lightweight linear heads on top of the shared encoder
        self.span_head    = nn.Linear(hidden, 2)
        self.center_head  = nn.Linear(hidden, num_centers)

        self.num_centers  = num_centers
        self.lambda_kl    = lambda_kl
        self.kl_loss      = nn.KLDivLoss(reduction="batchmean")
        self.ce           = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        token_type_ids:   Optional[torch.Tensor] = None,
        start_positions:  Optional[torch.Tensor] = None,
        end_positions:    Optional[torch.Tensor] = None,
        soft_labels:      Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass.

        During training supply start_positions, end_positions, and soft_labels.
        During inference only input_ids + attention_mask are required.

        Returns
        -------
        dict with:
            loss           (training only)
            start_logits   (seq_len,)
            end_logits     (seq_len,)
            center_logits  (K,)
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        seq            = out.last_hidden_state    # (B, L, H)
        cls            = seq[:, 0, :]             # (B, H)

        span_out       = self.span_head(seq)      # (B, L, 2)
        start_logits   = span_out[:, :, 0]        # (B, L)
        end_logits     = span_out[:, :, 1]        # (B, L)
        del span_out, seq

        center_logits  = self.center_head(cls)    # (B, K)
        center_log_p   = F.log_softmax(center_logits, dim=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            # Span extraction loss
            l_qa = (
                self.ce(start_logits, start_positions) +
                self.ce(end_logits,   end_positions)
            ) / 2.0

            # KL distillation loss (only when soft labels provided)
            l_kl = torch.tensor(0.0, device=l_qa.device)
            if soft_labels is not None:
                sl   = soft_labels.to(l_qa.device).clamp(min=1e-8)
                l_kl = self.kl_loss(center_log_p, sl)

            loss = self.lambda_kl * l_kl + (1.0 - self.lambda_kl) * l_qa

        return {
            "loss":          loss,
            "start_logits":  start_logits,
            "end_logits":    end_logits,
            "center_logits": center_logits,
        }


# ── Custom Trainer ───────────────────────────────────────────────────────────

class CenterDistillTrainer(Trainer):
    """
    HuggingFace Trainer subclass that passes soft_labels from the batch
    to the model's forward() method.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft_labels = inputs.pop("soft_labels", None)
        outputs     = model(soft_labels=soft_labels, **inputs)
        loss        = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# ── Training helper ──────────────────────────────────────────────────────────

def build_training_args(
    output_dir: str,
    epochs: int        = 4,
    batch_size: int    = 8,
    grad_accum: int    = 4,
    lr: float          = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int          = 42,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = warmup_ratio,
        weight_decay                = weight_decay,
        fp16                        = True,
        gradient_checkpointing      = True,
        dataloader_pin_memory       = True,
        save_strategy               = "epoch",
        logging_steps               = 20,
        seed                        = seed,
        report_to                   = "none",
    )


# ── HF-compatible patch ───────────────────────────────────────────────────────

def patch_to_hf_qa_model(
    cd_model: CenterDistillModel,
    base_model_path: str,
    tokenizer,
    save_path: str,
) -> None:
    """
    Extract the encoder + span head from a trained CenterDistillModel and
    save them as a standard HuggingFace QA model so that pipeline() works.

    This is needed for evaluate_qa() in data.py.
    """
    from transformers import AutoModelForQuestionAnswering

    hf_model = AutoModelForQuestionAnswering.from_pretrained(base_model_path)

    # Copy encoder weights (pooler mismatch is expected and harmless)
    hf_model.roberta.load_state_dict(
        cd_model.encoder.state_dict(), strict=False
    )
    # Copy span head
    hf_model.qa_outputs.weight = nn.Parameter(cd_model.span_head.weight.clone())
    hf_model.qa_outputs.bias   = nn.Parameter(cd_model.span_head.bias.clone())

    hf_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ HF-compatible model saved → {save_path}")
