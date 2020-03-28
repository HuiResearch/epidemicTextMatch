# -*- coding: utf-8 -*-
"""
File Name：     predict_utils
date：          2020/3/26
author:        'HuangHui'
"""
from transformers import InputExample, InputFeatures
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset
from collections import Counter
import csv


def load_data(filename):
    datas = pd.read_csv(filename).values.tolist()
    return datas


def create_examples(filename):
    datas = pd.read_csv(filename).values.tolist()
    examples = []
    for i, data in enumerate(datas):
        guid = data[0]
        text_a = data[2].strip()
        text_b = data[3].strip()
        examples.append(
            InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=None
            )
        )
    return examples


def create_features(examples, tokenizer, max_len):
    features = []
    pad_on_left = False
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0
    mask_padding_with_zero = True
    for example in tqdm(examples, desc='convert examples to features'):
        inputs = tokenizer.encode_plus(example.text_a, example.text_b,
                                       add_special_tokens=True, max_length=max_len)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_len - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_len, "Error with input length {} vs {}".format(len(input_ids), max_len)
        assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(
            len(attention_mask), max_len
        )
        assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(
            len(token_type_ids), max_len
        )
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=None
            )
        )
    return features


def create_dataset(filename, tokenizer, max_len):
    examples = create_examples(filename)
    features = create_features(examples, tokenizer, max_len)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_token_type_ids
    )
    ids = [example.guid for example in examples]
    return dataset, ids


def mean(logits):
    if len(logits) == 1:
        return logits[0]
    res = None
    for logit in logits:
        if res is None:
            res = logit
        else:
            res += logit
    res = res / len(logits)
    return res


def vote(predictions):
    '''
    投票融合方法
    :param predictions:
    :return:
    '''
    if len(predictions) == 1:  # 没有多个预测结果就直接返回第一个结果
        return predictions[0]
    result = []
    num = len(predictions[0])
    for i in range(num):
        temp = []
        for pred in predictions:
            temp.append(pred[i])
        counter = Counter(temp)
        result.append(counter.most_common()[0][0])
    return result


def write_result(filename, ids, predictions):
    with open(filename, 'w', encoding='utf-8') as w:
        writer = csv.writer(w, delimiter=",")
        writer.writerow(['id', 'label'])
        for id, pred in zip(ids, predictions):
            writer.writerow([id, int(pred)])
