# -*- coding: utf-8 -*-
"""T5-Large/mT5-Large model for Chinese academic keyphrase extraction.

The dataset in this repository is character-level Chinese text with optional
low-cost eye-tracking features.  A T5-style model is encoder-decoder by design;
for sequence labelling we use only the encoder states and place a lightweight
BIOES token-classification head on top.  The default checkpoint is mT5-Large
because its SentencePiece vocabulary is multilingual and covers Chinese text
far better than the original English-centric T5 vocabulary.
"""

import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, MT5EncoderModel, T5EncoderModel

from config import *
from evaluate import evaluate3, evaluate5, evaluate10


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_t5_encoder_class(model_name):
    """Select the correct encoder class for T5-family checkpoints."""
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")
    if model_type == "mt5":
        return MT5EncoderModel
    return T5EncoderModel


def _feature_indices():
    """Map Chinese eye-tracking feature names to dataset columns.

    Dataset feature order is FFD, FN, TFD.  Keeping this mapping explicit makes
    experiments reproducible and prevents accidental dimensional mismatches in
    the classifier when switching feature combinations.
    """
    mapping = {
        "no feature": [],
        "FFD": [0],
        "FN": [1],
        "TFD": [2],
        "FFD+FN": [0, 1],
        "FN+TFD": [1, 2],
        "FFD+TFD": [0, 2],
        "FFD+FN+TFD": [0, 1, 2],
    }
    return mapping.get(t5_feature, mapping.get(feature, []))


def load_data(data_path):
    """Load repository JSON data as Chinese character sequences.

    Each item is [character, FFD, FN, TFD, tag].  We keep characters separated
    and later pass them to the tokenizer with ``is_split_into_words=True`` so
    SentencePiece sub-tokens can be aligned back to the original Chinese
    character labels.
    """
    data_file = json.load(open(data_path, "r", encoding="utf-8"))
    texts, features, tags = [], [], []

    for items in data_file.values():
        chars, char_features, char_tags = [], [], []
        for item in items:
            chars.append(item[0])
            char_features.append([float(value) for value in item[1:-1]])
            char_tags.append(item[-1])
        texts.append(chars)
        features.append(char_features)
        tags.append(char_tags)

    return texts, features, tags


def align_label(tokenizer, chars, labels, features):
    """Align character-level labels/features to T5 SentencePiece tokens."""
    encoded = tokenizer(
        chars,
        max_length=t5_max_length,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    token_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    selected_features = _feature_indices()
    empty_features = [0.0] * len(selected_features)
    aligned_labels, aligned_features = [], []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(tag2ids["[PAD]"])
            aligned_features.append(empty_features)
        elif word_idx != previous_word_idx:
            aligned_labels.append(tag2ids.get(labels[word_idx], tag2ids["O"]))
            aligned_features.append([features[word_idx][idx] for idx in selected_features])
        else:
            if t5_label_all_tokens:
                aligned_labels.append(tag2ids.get(labels[word_idx], tag2ids["O"]))
                aligned_features.append([features[word_idx][idx] for idx in selected_features])
            else:
                aligned_labels.append(tag2ids["[PAD]"])
                aligned_features.append(empty_features)
        previous_word_idx = word_idx

    return encoded["input_ids"][0], encoded["attention_mask"][0], tokens, aligned_features, aligned_labels


class T5KeywordDataset(Dataset):
    """Dataset that preserves Chinese character boundaries for T5 tokenization."""

    def __init__(self, texts, old_features, tags, tokenizer):
        self.texts = texts
        self.old_features = old_features
        self.tags = tags
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.tokens = []
        self.features = []
        self.labels = []

    def encode(self):
        for idx in tqdm(range(len(self.texts)), desc="Encoding Chinese characters for T5"):
            input_ids, attention_mask, tokens, features, labels = align_label(
                self.tokenizer, self.texts[idx], self.tags[idx], self.old_features[idx]
            )
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.tokens.append(tokens)
            self.features.append(features)
            self.labels.append(labels)

        self.input_ids = torch.stack(self.input_ids)
        self.attention_masks = torch.stack(self.attention_masks)
        self.features = torch.tensor(np.array(self.features, dtype=np.float32), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels, dtype=np.int64), dtype=torch.long)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.tokens[idx],
            self.features[idx],
            self.labels[idx],
        )

    def __len__(self):
        return len(self.input_ids)


class T5LargeNerModel(nn.Module):
    """Encoder-only T5-Large token classifier with eye-tracking features."""

    def __init__(self, num_labels, model_name, feature_dim):
        super().__init__()
        encoder_class = _resolve_t5_encoder_class(model_name)
        self.encoder = encoder_class.from_pretrained(model_name)
        self.dropout = nn.Dropout(t5_dropout_value)
        self.feature_norm = nn.LayerNorm(feature_dim) if feature_dim > 0 else None
        hidden_size = self.encoder.config.d_model
        self.classifier = nn.Linear(hidden_size + feature_dim, num_labels)

    def forward(self, input_ids, attention_mask, extra_features):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(encoder_outputs.last_hidden_state)

        if self.feature_norm is not None:
            normalized_features = self.feature_norm(extra_features)
            sequence_output = torch.cat((sequence_output, normalized_features), dim=-1)

        return self.classifier(sequence_output)


def tag_convert(raw_tags, words_set, poss=None):
    """Convert BIOES token predictions back to keyword strings."""
    true_tags = []
    tag_array = raw_tags.detach().cpu().numpy() if torch.is_tensor(raw_tags) else raw_tags

    for i in range(tag_array.shape[0]):
        kw_list = []
        nkw_list = ""
        for j in range(len(tag_array[i])):
            item = int(tag_array[i][j])
            if item == tag2ids["[PAD]"]:
                continue
            if poss is not None and j in poss[i]:
                continue
            token = str(words_set[j][i]).replace("▁", "").replace("</s>", "").replace("<pad>", "")
            if not token:
                continue
            if item == tag2ids["O"]:
                nkw_list = ""
            elif item == tag2ids["S"]:
                if token not in kw_list:
                    kw_list.append(token)
            elif item == tag2ids["B"]:
                nkw_list = token
            elif item == tag2ids["I"]:
                nkw_list += token
            elif item == tag2ids["E"]:
                nkw_list += token
                if nkw_list and nkw_list not in kw_list:
                    kw_list.append(nkw_list)
                nkw_list = ""
        true_tags.append(kw_list)

    return true_tags


def _early_stop_score(metrics):
    """Return the configured validation metric used by early stopping."""
    metric_name = t5_early_stop_metric.upper()
    if metric_name not in metrics:
        raise ValueError(
            f"Unsupported t5_early_stop_metric={t5_early_stop_metric!r}. "
            "Choose one of: F3, F5, F10."
        )
    return metrics[metric_name]


def T5Large(train_path=train_path, test_path=test_path, vocab_path=None):
    """Train and evaluate T5-Large/mT5-Large for Chinese keyphrase extraction."""
    del vocab_path
    logging.info("Start T5-Large training with checkpoint: %s", t5_model_name)
    logging.info("T5 feature setting: %s", t5_feature)

    train_texts, train_features, train_tags = load_data(train_path)
    test_texts, test_features, test_tags = load_data(test_path)

    tokenizer = AutoTokenizer.from_pretrained(t5_model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("T5Large requires a fast tokenizer to align Chinese character labels with word_ids().")

    train_dataset = T5KeywordDataset(train_texts, train_features, train_tags, tokenizer)
    train_dataset.encode()
    test_dataset = T5KeywordDataset(test_texts, test_features, test_tags, tokenizer)
    test_dataset.encode()

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=t5_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=t5_eval_batch_size)

    model = T5LargeNerModel(
        num_labels=len(tag2ids),
        model_name=t5_model_name,
        feature_dim=len(_feature_indices()),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=t5_lr, weight_decay=t5_weight_decay)
    loss_fn = CrossEntropyLoss(ignore_index=tag2ids["[PAD]"]).to(device)

    best_values = {
        "epoch3": 0,
        "epoch5": 0,
        "epoch10": 0,
        "P3": 0.0,
        "R3": 0.0,
        "F3": 0.0,
        "P5": 0.0,
        "R5": 0.0,
        "F5": 0.0,
        "P10": 0.0,
        "R10": 0.0,
        "F10": 0.0,
    }
    fin_targets, fin_prediction = [], []
    best_early_stop_score = -float("inf")
    early_stop_rounds = 0

    for epoch in tqdm(range(t5_epochs), desc="Training T5-Large"):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            input_ids, attention_masks, _, features, tags = batch
            pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))
            loss = loss_fn(pred_tags.view(-1, len(tag2ids)), tags.to(device).view(-1))
            loss = loss / t5_gradient_accumulation_steps
            loss.backward()

            if (step + 1) % t5_gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t5_max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * t5_gradient_accumulation_steps

        avg_loss = total_loss / max(len(train_dataloader), 1)
        print("avg_loss: %.4f" % avg_loss)
        logging.info("epoch: %s", epoch)
        logging.info("avg_loss: %.4f", avg_loss)

        model.eval()
        kw_true, kw_pred = [], []
        for batch in test_dataloader:
            input_ids, attention_masks, tokens, features, tags = batch
            with torch.no_grad():
                pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))
                pred_tags = F.softmax(pred_tags, dim=-1)
                pred_tags = torch.argmax(pred_tags, dim=-1)

            poss = []
            for row in tags:
                pos = []
                for j in range(len(row)):
                    if int(row[j]) == tag2ids["[PAD]"]:
                        pos.append(j)
                poss.append(pos)

            kw_true.extend(tag_convert(tags, tokens))
            kw_pred.extend(tag_convert(pred_tags, tokens, poss))

        P3, R3, F3 = evaluate3(kw_true, kw_pred)
        P5, R5, F5 = evaluate5(kw_true, kw_pred)
        P10, R10, F10 = evaluate10(kw_true, kw_pred)
        current_metrics = {"F3": F3, "F5": F5, "F10": F10}

        if epoch == 0 or F3 > best_values["F3"]:
            best_values.update({"F3": F3, "P3": P3, "R3": R3, "epoch3": epoch})
            fin_targets = kw_pred
            fin_prediction = kw_true
        if epoch == 0 or F5 > best_values["F5"]:
            best_values.update({"F5": F5, "P5": P5, "R5": R5, "epoch5": epoch})
        if epoch == 0 or F10 > best_values["F10"]:
            best_values.update({"F10": F10, "P10": P10, "R10": R10, "epoch10": epoch})

        if t5_early_stop:
            early_stop_score = _early_stop_score(current_metrics)
            if early_stop_score > best_early_stop_score + t5_early_stop_min_delta:
                best_early_stop_score = early_stop_score
                early_stop_rounds = 0
            else:
                early_stop_rounds += 1
                logging.info(
                    "T5 early stop counter: %s/%s, %s=%.6f, best=%.6f",
                    early_stop_rounds,
                    t5_early_stop_patience,
                    t5_early_stop_metric,
                    early_stop_score,
                    best_early_stop_score,
                )
            if early_stop_rounds >= t5_early_stop_patience:
                logging.info(
                    "Early stopping T5-Large at epoch %s because %s did not improve by %.6f for %s epochs.",
                    epoch,
                    t5_early_stop_metric,
                    t5_early_stop_min_delta,
                    t5_early_stop_patience,
                )
                print(
                    "Early stopping T5-Large at epoch",
                    epoch,
                    "because",
                    t5_early_stop_metric,
                    "did not improve."
                )
                break

    with open(save_path, mode="a+", encoding="utf-8") as f:
        for keywords in fin_prediction:
            f.write(",".join(keywords) + "\n")
        f.write("----------------------\n")
        for keywords in fin_targets:
            f.write(",".join(keywords) + "\n")

    return (
        best_values["epoch3"],
        best_values["epoch5"],
        best_values["epoch10"],
        best_values["P3"],
        best_values["R3"],
        best_values["F3"],
        best_values["P5"],
        best_values["R5"],
        best_values["F5"],
        best_values["P10"],
        best_values["R10"],
        best_values["F10"],
    )
