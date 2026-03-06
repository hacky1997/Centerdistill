"""
data.py — MLQA data loading and tokenisation helpers for CenterDistill.

All three language pair loaders return lists of dicts with keys:
    id, question, context, answers

The `make_tokenise_fn` factory produces a HuggingFace-compatible
tokenisation function for use with Trainer.
"""

import json
from typing import List, Dict, Tuple


# ── Raw JSON helpers ─────────────────────────────────────────────────────────

def _flatten(raw_data: list) -> List[Dict]:
    """Flatten SQuAD-format JSON into a flat list of QA examples."""
    out = []
    for article in raw_data:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                out.append({
                    "id":       qa["id"],
                    "question": qa["question"],
                    "context":  ctx,
                    "answers":  qa["answers"],
                })
    return out


def _load_split(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return _flatten(json.load(f)["data"])


# ── Language-pair loaders ────────────────────────────────────────────────────

def load_en_en(
    mlqa_root: str = "MLQA_V1",
    train_val_split: float = 0.8,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Return (train, val, test) for the English–English split.

    train / val are carved from the dev set (80 / 20).
    test  is the full en-en test file.
    """
    dev  = _load_split(f"{mlqa_root}/dev/dev-context-en-question-en.json")
    test = _load_split(f"{mlqa_root}/test/test-context-en-question-en.json")
    cut  = int(len(dev) * train_val_split)
    return dev[:cut], dev[cut:], test


def load_en_es(mlqa_root: str = "MLQA_V1") -> Tuple[List[Dict], List[Dict]]:
    """
    Return (dev, test) for English context + Spanish question.
    Primary cross-lingual evaluation pair.
    """
    dev  = _load_split(f"{mlqa_root}/dev/dev-context-en-question-es.json")
    test = _load_split(f"{mlqa_root}/test/test-context-en-question-es.json")
    return dev, test


def load_en_de(mlqa_root: str = "MLQA_V1") -> Tuple[List[Dict], List[Dict]]:
    """
    Return (dev, test) for English context + German question.
    Secondary cross-lingual evaluation pair.
    """
    dev  = _load_split(f"{mlqa_root}/dev/dev-context-en-question-de.json")
    test = _load_split(f"{mlqa_root}/test/test-context-en-question-de.json")
    return dev, test


# ── Tokenisation ─────────────────────────────────────────────────────────────

def make_tokenise_fn(tokenizer, max_len: int = 384, stride: int = 128):
    """
    Return a HuggingFace Trainer-compatible tokenisation function.

    The returned function maps a single example dict to tokenised features
    with `start_positions` and `end_positions` included.
    """
    def tokenise(example: Dict) -> Dict:
        enc = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=max_len,
            stride=stride,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offsets = enc.pop("offset_mapping")
        ans     = example["answers"][0]
        char_s  = ans["answer_start"]
        char_e  = char_s + len(ans["text"])

        tok_start, tok_end = 0, 0
        for i, (s, e) in enumerate(offsets):
            if s <= char_s < e:
                tok_start = i
            if s < char_e <= e:
                tok_end = i
                break

        enc["start_positions"] = tok_start
        enc["end_positions"]   = tok_end
        return enc

    return tokenise


def make_tokenise_fn_with_soft_labels(
    tokenizer,
    soft_label_map: Dict[str, list],
    max_len: int = 384,
    stride: int = 128,
):
    """
    Extended tokenise function that also attaches a soft_labels tensor
    (teacher center distribution) keyed by example id.

    Parameters
    ----------
    soft_label_map : dict mapping example_id → list[float] (length K)
    """
    import torch
    base_fn = make_tokenise_fn(tokenizer, max_len, stride)

    def tokenise(example: Dict) -> Dict:
        enc = base_fn(example)
        sl  = soft_label_map.get(example["id"])
        if sl is not None:
            enc["soft_labels"] = sl
        return enc

    return tokenise


# ── QA evaluation helper ─────────────────────────────────────────────────────

def evaluate_qa(
    model_path: str,
    examples: List[Dict],
    n: int = 1000,
    device: int = 0,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Run a HuggingFace QA pipeline on the first `n` examples.

    Returns
    -------
    (scores, predictions, references)
        scores       : dict with 'exact_match' and 'f1'
        predictions  : list of {id, prediction_text}
        references   : list of {id, answers: {text, answer_start}}
    """
    import torch
    import evaluate as hf_evaluate
    from transformers import pipeline

    squad = hf_evaluate.load("squad")
    qa    = pipeline("question-answering", model=model_path, device=device)

    preds, refs = [], []
    for ex in examples[:n]:
        try:
            ans = qa(question=ex["question"], context=ex["context"])["answer"]
        except Exception:
            ans = ""
        preds.append({"id": ex["id"], "prediction_text": ans})
        refs.append({
            "id": ex["id"],
            "answers": {
                "text":         [a["text"]         for a in ex["answers"]],
                "answer_start": [a["answer_start"] for a in ex["answers"]],
            },
        })

    scores = squad.compute(predictions=preds, references=refs)
    del qa
    torch.cuda.empty_cache()
    return scores, preds, refs
