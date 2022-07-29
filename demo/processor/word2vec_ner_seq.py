import copy
import json

import gensim
import torch
import logging
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_len, label_ids):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    batch应该是（序列、目标、长度）元组的列表。。。
    
    Returns a padded tensor of sequences sorted from longest to shortest,
    返回从最长到最短排序的序列的填充张量，
    """
    all_input_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids,  all_lens, all_labels

def convert_examples_to_features(examples, label_list, key_index_dict,  max_seq_length,
                                 pad_label_ids=0,):
    """
    通过word2vec 的词典，将文本转化为ids
    """
    #print(label_list)

    none_zero_sample_count = 0
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a, list):
            example.text_a = " ".join(example.text_a)

        tokens = example.text_a.lower().split(" ")

        #print(example.labels)
        #print(label_map)

        label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        if len(tokens) > max_seq_length:
            tokens = tokens[: max_seq_length]
            label_ids = label_ids[: max_seq_length]

        input_ids = [key_index_dict[token] if token in key_index_dict.keys()
                     else key_index_dict["none"] for token in tokens]
        input_len = len(label_ids)

        padding_length = max_seq_length - len(input_ids)

        input_ids += [key_index_dict["none"]] * padding_length
        label_ids += [pad_label_ids] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_len=input_len, label_ids=label_ids))
    print(examples[0].guid.split("-")[0], " dataset has samples ", none_zero_sample_count)
    return features