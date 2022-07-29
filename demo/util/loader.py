
def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)

    大写特征：
     0 = 低上限
     1 = 全部大写
     2 = 第一个字母大写
     3 = 一个大写字母（不是第一个字母）
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    准备一个句子进行评估。
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }

#########################目前只用了下面这个函数
def prepare_dataset(sentences, word_to_id, char_to_id, label_ids_list, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    准备数据集。 返回包含以下内容的字典列表列表：
         - 词索引
         - 字字符索引
         - 标签索引
    """
    def f(x): return x.lower() if lower else x
    data = []
    index=0
    for s in sentences:
        str_words = [w for w in s]
        #print("str_words")#['The', 'admin@338', 'has', 'largely']
        #print(str_words)
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set（跳过不在训练集中的字符）
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        #caps = [cap_feature(w) for w in str_words]
        
        #tags = [tag_to_id[w[-1]] for w in s]
        #print("tags")#[0, 1, 0, 0]
        #print(tags)
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags':label_ids_list[index]
            #'caps': caps,
            #'tags': tags,
        })
        index=index+1
        '''{'str_words': ['The', 'admin@338', 'has', 'largely'], 
        'words': [0, 500, 21, 683], 
        'chars': [[24, 11, 0], [2, 10, 13, 3, 4, 77, 43, 43, 47], [11, 2, 6], [8, 2, 7, 15, 0, 8, 17]], 
        'caps': [2, 0, 0, 0], 
        'tags': [0, 1, 0, 0]}'''
        
    return data


