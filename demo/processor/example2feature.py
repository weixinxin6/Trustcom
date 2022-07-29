#单词级没有区别大小写，字符级区分大小写

#把example(包括单词和标签)转化为对应的向量
import copy
import json

import gensim
import torch
import logging
import sys
project_root_path = "F:/A2_postgraduate/pt_model/demo_1/demo"
sys.path.append(project_root_path)
from util.utils_bilstm_crf import create_dico, create_mapping, zero_digits
from util.loader import prepare_dataset

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
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids,  all_lens, all_labels


def word_mapping(dico):#全部小写
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    创建单词映射，按频率排序。
    """
    #print("单个句子：")
    #print(sentences[0])#['\ufeffthe', 'admin@338', 'has', 
    #dico = create_dico(sentences)

    #dico['<UNK>'] = 10000000
    #print("字典")
    #print(dico)#{'The': 2, 'admin@338': 2, 'has': 2, 'largely': 2, '<UNK>': 10000000}

    word_to_id, id_to_word = create_mapping(dico)
    '''print ("Found %i unique words (%i in total)" % ( #Found 7986 unique words (140353 in total)
        len(dico), sum(len(x) for x in sentences)
    ))'''
    #print("word_to_id:")#{'<UNK>': 0, 'The': 1, 'admin@338': 2, 'has': 3, 'largely': 4}
    #print(word_to_id)#每一个word都有一个唯一的ID

    return word_to_id, id_to_word



def char_mapping(sentences):#区分大小写
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    创建字典和字符映射，按频率排序。
    """
    '''
    words=[]
    sens=[]
    word1=['The', 'O']
    word2=['admin@338', 'B-HackOrg']
    word3=['has', 'O']
    word4=['largely', 'O']
    words.append(word1)
    words.append(word2)   
    words.append(word3)   
    words.append(word4)
    sens.append(words)
    sens.append(words)
    sens.append(words)
    print("sens")
    print(sens[0])#[['The', 'O'], ['admin@338', 'B-HackOrg'], ['has', 'O'], ['largely', 'O']]

    chars = ["".join([w[0] for w in s]) for s in sens]
    print("char")
    print(chars[0])#Theadmin@338haslargely
    '''
    chars = ["".join([w for w in s]) for s in sentences]
    #print("char")
    #print(chars[0])#Theadmin@338haslargelytargetedorganization

    dico = create_dico(chars)
    #print("dico")
    #print(dico)

    char_to_id, id_to_char = create_mapping(dico)
    print ("Found %i unique characters" % len(dico))#131
    #print("char_to_id")
    #print(char_to_id)

    return dico, char_to_id, id_to_char


'''def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    创建字典和标签映射，按频率排序。
    """
    words=[]
    sens=[]
    word1=['The', 'O']
    word2=['admin@338', 'B-HackOrg']
    word3=['has', 'O']
    word4=['largely', 'O']
    words.append(word1)
    words.append(word2)   
    words.append(word3)   
    words.append(word4)
    sens.append(words)
    sens.append(words)
    sens.append(words)
    print("sens")

    tags = [[word[-1] for word in s] for s in sentences]#sentences/sens
    print("tag")
    print(tags[0])#['O', 'B-HackOrg', 'O', 'O']

    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print ("Found %i unique named entity tags" % len(dico))
    print(dico)
    #print(tag_to_id)
    #print(id_to_tag)
    return dico, tag_to_id, id_to_tag'''


def convert_examples_to_features(examples, label_list,  max_seq_length,
                                 pad_label_ids=0,):

    sentences = []
    sentence = []
    tokenList=[]
    tokenListNoLower=[]

    label_ids_list=[]

    """
    通过word2vec 的词典，将文本转化为ids
    """

    none_zero_sample_count = 0
    label_map = {label: i for i, label in enumerate(label_list)}
    #print("label_map")# label_map就是tag_to_id
    #print(label_map)#{'O': 0, 'B-HackOrg': 1, 'I-HackOrg': 2,...
    features = []

    for (ex_index, example) in enumerate(examples):
        #print("ex_index, example")
        #print(ex_index)# 0,1,2  索引
        #print(example)# examples[ex_index],单词example.text_a和标签example.labels
        #print(example.text_a)#数组
        if ex_index % 10000 == 0:
            print("info")
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a, list):#判断一个对象(example)是否是一个已知的类型（list）
            example.text_a = " ".join(example.text_a)#以空格作为分隔符，将example.text_a所有的元素合并成一个新的字符串，即每句话的单词之间用空格隔开。

        #print("example.text_a")
        #print(example.text_a)#一句话变成了一个字符串，单词之间用空格隔开
        sentence=example.text_a
        sentences.append(sentence)

        tokens = example.text_a.lower().split(" ")#lower() 函数的作用是把一个字符串中所有大写形式的字符变为小写形式
        tokens_no_lower=example.text_a.split(" ")#区分大小写
        #print("tokens")
        #print(tokens)#['\ufeffthe', 'admin@338', 'has', 'largely', 'targeted',

        for x in example.labels:
            if(x==''):
                print("数据错误")
                print(example)

        label_ids = [label_map[x] for x in example.labels]
        #print("label_ids")#把单词对应的标签转化为数字
        #print(label_ids)#[0, 1, 0, 0, 0, 0, 0, 0, 15, 0, 15,
        label_ids_list.append(label_ids)

        # Account for [CLS] and [SEP] with "- 2".
        if len(tokens) > max_seq_length:#如果一句话长度大于max_seq_length个词，则只取前max_seq_length个词
            tokens = tokens[: max_seq_length]
            label_ids = label_ids[: max_seq_length]

        if len(tokens_no_lower) > max_seq_length:#如果一句话长度大于max_seq_length个词，则只取前max_seq_length个字词
            tokens_no_lower = tokens_no_lower[: max_seq_length]
            label_ids = label_ids[: max_seq_length]
        
        tokenList.append(tokens)
        tokenListNoLower.append(tokens_no_lower)
###############################################################################################
        #上面是把标签转化为ids,下面是把单词转化为ids

        '''input_ids = [key_index_dict[token] if token in key_index_dict.keys()
                     else key_index_dict["none"] for token in tokens]
        input_len = len(label_ids)

        padding_length = max_seq_length - len(input_ids)

        input_ids += [key_index_dict["none"]] * padding_length
        label_ids += [pad_label_ids] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_len=input_len, label_ids=label_ids))
    '''
    #创建字典
    dico = create_dico(tokenList)
    dico['<UNK>'] = 10000000
    print ("Found %i unique words (%i in total)" % ( #Found 7986 unique words (140353 in total)
        len(dico), sum(len(x) for x in tokenList)
    ))

    # Create a dictionary / mapping of words(创建字典/单词映射)
    word_to_id, id_to_word = word_mapping(dico)
    
    # Create a dictionary and a mapping for words / POS tags / tags
    ##为单词/词性标签/标签创建字典和映射
    dico_chars, char_to_id, id_to_char = char_mapping(tokenListNoLower)#若想把大写换为小写，传参为tokenList
    #dico_tags, tag_to_id, id_to_tag = tag_mapping(sentences)

    lower=0
    data = prepare_dataset(
        tokenList, word_to_id, char_to_id,label_ids_list,lower #label_map==tag_to_id
    )

    #print("data")
    #print(data[1])

    for d in data:
        input_len = len(d.get("tags"))

        input_ids = d.get("words")
        label_ids = d.get("tags")

        if len(label_ids) > max_seq_length:#如果一句话长度大于max_seq_length个词，则只取前max_seq_length个词
            label_ids = label_ids[: max_seq_length]

        pad_label_ids = 0
        padding_length = max_seq_length - len(input_ids)
        if(padding_length>0):     
             input_ids += [pad_label_ids] * padding_length
             label_ids += [pad_label_ids] * padding_length       
        
        #print("max_seq_length:")
        #print(max_seq_length) #128

        #print("len(input_ids)")
        #print(len(input_ids))  #128

        #print("len(label_ids)")
        #print(len(label_ids))  #128

        #assert len(input_ids) == max_seq_length
        #assert len(label_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_len=input_len, label_ids=label_ids))

    #print("features")
    #print(features[1])

    #print(examples[0].guid.split("-")[0])#train
    #print(examples[0].guid.split("-")[0], " dataset has samples ", none_zero_sample_count)
    #print("词典大小：")
    #print(len(dico))#7986
    return features


