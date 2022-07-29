""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
import numpy as np
from processor.utils_ner import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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

class InputFeatures_Resume(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_len, resume_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len
        self.resume_mask = resume_mask
        self.label_ids = label_ids

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
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens,  all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens

def collate_fn_for_resume(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_resume_masks, all_labels = map(torch.stack,
                                                                                                        zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_resume_masks = all_resume_masks[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_resume_masks, all_labels

def collate_fn_for_resume_raw(batch):
    return batch


def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels




def tokenize_and_preserve_labels_for_resume(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    resume_mask = []  # 2 表示 开头，1 表示 中间，0 表示不是子词
    labels = []
    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        if n_subwords <= 1:
            resume_mask.append(0)
            labels.append(label)
            tokenized_sentence.append(word)
        else:
            resume_mask.append(2)
            resume_mask.extend([1] * (n_subwords - 1))
            labels.extend([label] * (n_subwords))
        # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
    assert len(tokenized_sentence) == len(resume_mask)
    assert len(tokenized_sentence) == len(labels)
    return tokenized_sentence, resume_mask, labels


def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True, language="ch"):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    none_zero_sample_count = 0
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    #index=0
    #print(examples[5252])
    
    for (ex_index, example) in enumerate(examples):
        #index=index+1
        #print(index)

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a,list):
            example.text_a = " ".join(example.text_a)
        if language == "en":
                tokens, labels = tokenize_and_preserve_labels(tokenizer, example.text_a.split(" "), example.labels)
                
                for x in labels:
                    if(x=="E-APT\xa0"):
                        print("数据错误")
                        print(example)
                
                label_ids = [label_map[x] for x in labels]
        else:
            tokens = tokenizer.tokenize(example.text_a)
            label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        # np_label_ids = np.array(label_ids)
        # if np_label_ids.sum() == 0:
        #     continue
        # none_zero_sample_count += 1
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))
    print(examples[0].guid.split("-")[0], " dataset has samples ", none_zero_sample_count)
    return features

def convert_examples_to_features_for_resume(
        text_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        label_list=None
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    label_map = {label: i for i, label in enumerate(label_list)}
    for temp_data in text_list:
        if temp_data.guid == "train-11":
            print("BREAK")
        temp_text = temp_data.text_a
        temp_label = temp_data.labels
        tokens, resume_mask, labels = tokenize_and_preserve_labels_for_resume(tokenizer,
                                                                              temp_text,
                                                                              temp_label)
        label_ids = [label_map[x] for x in labels]
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            resume_mask = resume_mask[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [label_map['O']]
        resume_mask += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [label_map["O"]] + label_ids
        resume_mask = [0] + resume_mask
        segment_ids = [cls_token_segment_id] + segment_ids


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids += [pad_token] * padding_length
        resume_mask += [0] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(resume_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        features.append(InputFeatures_Resume(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                      input_len=input_len, resume_mask=resume_mask, label_ids=label_ids))
    return features




class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position','B-scene',"I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position','I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene','O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CyberProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B-RT", "I-RT", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-SW", "I-SW", "B-VUL_ID",
                "I-VUL_ID", "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 9619:
                print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


class APTProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "train_data.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "dev_data.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "test_data.json")), "test")

    def get_labels(self):
        """See base class."""
        # return [
        #         'O', 'B-Threat_actor', 'I-Threat_actor', 'E-Threat_actor',  'S-Threat_actor',
        #         'B-Threat_actor_aliases', 'I-Threat_actor_aliases', 'E-Threat_actor_aliases', 'S-Threat_actor_aliases',
        #         'B-Reference_word', 'I-Reference_word', 'E-Reference_word', 'S-Reference_word',
        #          'B-Malware_tool', 'I-Malware_tool',  'E-Malware_tool', 'S-Malware_tool',
        #         'B-Target', 'I-Target', 'E-Target', 'S-Target', 'B-Software', 'I-Software', 'E-Software', 'S-Software',
        #         'B-Industry', 'I-Industry', 'E-Industry', 'S-Industry',
        #         'B-Geo_location', 'I-Geo_location', 'E-Geo_location', 'S-Geo_location']
        return ['O', 'B-Threat_actor', 'I-Threat_actor', 'B-Software', 'I-Software', 'B-Geo_location', 'I-Geo_location',
                'B-Target', 'I-Target', 'B-Threat_actor_aliases', 'I-Threat_actor_aliases',
                'B-Reference_word', 'I-Reference_word', 'B-Industry', 'I-Industry', 'B-Malware_tool', 'I-Malware_tool',
                "[START]", "[END]"]


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 9619:
                print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class WANG_NER(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "train_data.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "dev_data.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "test_data.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B-location", "I-location", "B-tool", "I-tool", "B-threatactor_name", "I-threatactor_name",
                "B-reference_word", "I-reference_word", "B-malware", "I-malware", "B-target",
                "I-target", "B-industry", "I-industry",  "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples



class WANGORI_NER(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "train_data.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "dev_data.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "test_data.json")), "test")

    def get_labels(self):
        """See base class."""
        return ['O', 'B-location', 'I-location', 'B-tool', 'I-tool', 'B-time', 'I-time', 'B-threatactor_name', 'I-threatactor_name',
                'B-attack_activity', 'I-attack_activity', 'B-reference_word', 'I-reference_word', 'B-malware',
                'I-malware', 'B-person', 'I-person', 'B-company', 'I-company',  'B-attack_goal',
                'I-attack_goal', 'B-industry', 'I-industry', 'B-government', 'I-government', 'B-threatactor_aliases',
                'I-threatactor_aliases', 'B-target_crowd', 'I-target_crowd', 'B-protocol', 'I-protocol', 'B-security_team',
                'I-security_team', 'B-vulnerability_cve','I-vulnerability_cve', 'B-sub_activity', 'I-sub_activity',
                'B-email_evil', 'I-email_evil', 'B-string', 'I-string',  'B-domain', 'I-domain', 'B-ip', 'I-ip',
                'B-sha2', 'I-sha2', 'B-sample_name', 'I-sample_name', 'B-os_name', 'I-os_name', 'B-counter_measure',
                'I-counter_measure', 'B-vul_aliases', 'I-vul_aliases',  'B-sample_function', 'I-sample_function',
                'B-domain_evil', 'I-domain_evil', 'B-program_language', 'I-program_language', 'B-ip_evil', 'I-ip_evil',
                'B-url_evil', 'I-url_evil', 'B-url', 'I-url', 'B-md5', 'I-md5', 'B-function', 'I-function', 'B-sha1',
                'B-encryption_algo', 'I-encryption_algo', "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class DNRTIProcessor(DataProcessor):
    """Processor for the english ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B-HackOrg", "I-HackOrg", "B-OffAct", "I-OffAct", "B-SamFile", "I-SamFile", "B-SecTeam", "I-SecTeam", "B-Time",
                "I-Time", "B-Way", "I-Way", "B-Tool", "I-Tool", "B-Idus", "I-Idus", "B-Org", "I-Org", "B-Area", "I-Area",
                "B-Purp", "I-Purp", "B-Exp", "I-Exp", "B-Features", "I-Features", "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 9619:
            #     print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class APTNERProcessor(DataProcessor):
    """Processor for the english ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B-APT", "I-APT", "E-APT","S-APT",
                "B-SECTEAM", "I-SECTEAM", "E-SECTEAM","S-SECTEAM",
                "B-IDTY", "I-IDTY", "E-IDTY","S-IDTY",
                "B-OS", "I-OS", "E-OS","S-OS",
                "B-EMAIL", "I-EMAIL", "E-EMAIL","S-EMAIL",
                "B-LOC", "I-LOC", "E-LOC","S-LOC",
                "B-TIME", "I-TIME", "E-TIME","S-TIME",
                "B-IP", "I-IP", "E-IP","S-IP",
                "B-DOM", "I-DOM", "E-DOM","S-DOM",
                "B-URL", "I-URL", "E-URL","S-URL",
                "B-PROT", "I-PROT", "E-PROT","S-PROT",
                "B-FILE", "I-FILE", "E-FILE","S-FILE",
                "B-TOOL", "I-TOOL", "E-TOOL","S-TOOL",
                "B-MD5", "I-MD5", "E-MD5","S-MD5",
                "B-SHA1", "I-SHA1", "E-SHA1","S-SHA1",
                "B-SHA2", "I-SHA2", "E-SHA2","S-SHA2",
                "B-MAL", "I-MAL", "E-MAL","S-MAL",
                "B-ENCR", "I-ENCR", "E-ENCR","S-ENCR",
                "B-VULNAME", "I-VULNAME", "E-VULNAME","S-VULNAME",
                "B-VULID", "I-VULID", "E-VULID","S-VULID",
                "B-ACT", "I-ACT", "E-ACT","S-ACT"]
        """return ["O", "B-APT", "I-APT", "E-APT","S-APT",
            "B-SECTEAM", "I-SECTEAM", "E-SECTEAM","S-SECTEAM",
            "B-IDTY", "I-IDTY", "E-IDTY","S-IDTY",
            "B-OS", "I-OS", "E-OS","S-OS",
            "B-EMAIL", "E-EMAIL","S-EMAIL",
            "B-LOC", "I-LOC", "E-LOC","S-LOC",
            "B-TIME", "I-TIME", "E-TIME","S-TIME",
            "B-IP","I-IP","E-IP","S-IP",
            "B-DOM","I-DOM","E-DOM","S-DOM",
            "B-URL", "I-URL", "E-URL","S-URL",
            "B-PROT", "I-PROT", "E-PROT","S-PROT",
            "B-FILE", "I-FILE", "E-FILE","S-FILE",
            "B-TOOL", "I-TOOL", "E-TOOL","S-TOOL",
            "S-MD5",
            "S-SHA1",
            "B-SHA2","E-SHA2","S-SHA2",
            "B-MAL", "I-MAL", "E-MAL","S-MAL",
            "B-ENCR", "E-ENCR","S-ENCR",
            "B-VULNAME", "I-VULNAME", "E-VULNAME","S-VULNAME",
            "B-VULID","S-VULID",
            "B-ACT", "I-ACT", "E-ACT","S-ACT"]"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 9619:
            #     print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CTIReportProcessor(DataProcessor):
    """Processor for the english ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["O",
                "B-malware.infosteal","I-malware.infosteal",
                "B-malware.backdoor","I-malware.backdoor",
                "B-url.cncsvr","I-url.cncsvr",
                "B-malware.unknown","I-malware.unknown",
                "B-hash","I-hash",
                "B-malware.drop","I-malware.drop",
                "B-url.normal","I-url.normal",
                "B-ip.unknown","I-ip.unknown",
                "B-url.unknown","I-url.unknown",
                "B-malware.ransom","I-malware.ransom",
                ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 9619:
            #     print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

# class AutoLabelProcessor(DataProcessor):
#     """Processor for the english ner data set."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_text(os.path.join(data_dir, "train_data.txt")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_text(os.path.join(data_dir, "dev_data.txt")), "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_text(os.path.join(data_dir, "test_data.txt")), "test")
#
#     def get_labels(self):
#         """See base class."""
#         return ["O", "B-edition", "I-edition", "B-update", "I-update", "B-application", "I-application",
#                 "B-version", "I-version", "B-vendor", "I-vendor", "B-hardware", "I-hardware", "B-relevant_term",
#                 "I-relevant_term", "B-programming_language", "B-method", "B-os", "I-os", "B-cve_id", "B-file",
#                 "B-language", "B-function", "B-parameter", "[START]", "[END]"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             # if i == 9619:
#             #     print("BREAK")
#             guid = "%s-%s" % (set_type, i)
#             text_a= line['words']
#             # BIOS
#             labels = line['labels']
#             examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
#         return examples

class AutoLabelProcessor(DataProcessor):
    """Processor for the english ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "train_data.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "dev_data.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "test_data.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B-edition", "I-edition", "B-update", "I-update", "B-application", "I-application",
                        "B-version", "I-version", "B-vendor", "I-vendor", "B-hardware", "I-hardware", "B-relevant_term",
                        "I-relevant_term", "B-programming_language", "B-method", "B-os", "I-os", "B-cve_id", "B-file",
                        "B-language", "B-function", "B-parameter", "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 9619:
            #     print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples



class VulinconProcessor(DataProcessor):
    """Processor for the english ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "train_data.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "dev_data.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "test_data.json")), "test")

    def get_labels(self):
        """See base class."""
        # return ["O", "B-SN", "I-SN", "B-SV", "I-SV", "[START]", "[END]"]
        return ["O", "B-ID", "I-ID", "B-ORG", "I-ORG", "B-PRO", "I-PRO", "B-VUL", "I-VUL",
                "B-VER", "I-VER", "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 9619:
            #     print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


class VultwitProcessor(DataProcessor):
    """Processor for the english ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "train_data.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "dev_data.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_list(os.path.join(data_dir, "test_data.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B-ID", "I-ID", "B-ORG", "I-ORG", "B-PRO", "I-PRO", "B-VUL", "I-VUL",
                "B-VER", "I-VER", "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 9619:
            #     print("BREAK")
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

#APTNERProcessor有两种返回值情况，一种是在标签集合中去掉标签没有出现过的；一种是保留全部预定义标签
ner_processors = {
    "dnrti": DNRTIProcessor,
    "aptner":APTNERProcessor,
    "autolabel": AutoLabelProcessor,
    "cti_report":CTIReportProcessor,##前四个实现了
    "cner": CnerProcessor,
    'cluener': CluenerProcessor,
    "cyber": CyberProcessor,
    #"autolabel": AutoLabelProcessor,
    "autolabel_general_w2v": AutoLabelProcessor,
    #"aptner": APTProcessor,
    "wang_ori": WANGORI_NER,
    "wangner": WANG_NER,
    "wangner_enbert": WANG_NER,
    "wangner_cybert": WANG_NER,
    "vulincon": VulinconProcessor,
    "vulincon_general_w2v": VulinconProcessor,
    "vultwit": VultwitProcessor,
    "vultwit_general_w2v": VultwitProcessor,

}
