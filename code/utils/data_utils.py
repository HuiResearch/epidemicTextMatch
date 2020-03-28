# -*- coding: utf-8 -*-
"""
File Name：     data_utils
date：          2020/3/26
author:        'HuangHui'
"""

import logging
import os
from transformers import InputFeatures, InputExample
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from functools import partial
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

logger = logging.getLogger(__name__)


class PairProcessor:
    def load_data(self, filename):
        datas = pd.read_csv(filename).values.tolist()
        return datas

    def get_labels(self):
        return ['0', '1']

    def get_examples(self, data_dir, set_type):
        file_map = {'train': 'train.csv',
                    'dev': 'dev.csv',
                    'test': 'test.example.csv'}
        file_name = os.path.join(data_dir, file_map[set_type])
        datas = self.load_data(file_name)
        examples = self.create_examples(datas, set_type)
        return examples

    def create_examples(self, datas, set_type):
        examples = []
        for i, data in enumerate(datas):
            guid = data[0]
            text_a = data[2].strip()
            text_b = data[3].strip()
            if set_type == 'test':
                label = None
            else:
                label = str(int(data[4]))
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label
                )
            )
        return examples


def classification_convert_example_to_feature(
        example,
        max_length=512,
        label_map=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        set_type='train'
):
    inputs = tokenizer.encode_plus(example.text_a,
                                   example.text_b,
                                   add_special_tokens=True, max_length=max_length)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length
    )
    if set_type != 'test':
        label = label_map[example.label]
    else:
        label = None

    return InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
    )


def multi_classification_convert_examples_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def multi_classification_convert_examples_to_dataset(
        examples,
        tokenizer,
        max_length=512,
        label_list=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        threads=10,
        set_type='train'
):
    label_map = dict(zip(label_list, range(len(label_list))))
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=multi_classification_convert_examples_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            classification_convert_example_to_feature,
            max_length=max_length,
            label_map=label_map,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
            mask_padding_with_zero=mask_padding_with_zero,
            set_type=set_type
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            )
        )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if set_type != 'test':
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_labels
        )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids
        )
    del features
    return dataset


def compute_metrics(y_true, y_pred, average='micro'):
    result = {}
    f1 = f1_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    result['f1'] = round(f1, 4)
    result['recall'] = round(recall, 4)
    result['precision'] = round(precision, 4)
    result['accuracy'] = round(accuracy, 4)
    return result


PROCESSORS = {
    'pair': PairProcessor
}
