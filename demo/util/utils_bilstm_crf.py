import os
import re
import codecs
import numpy as np
import theano

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    从项目列表创建项目字典。

    参数item_list是一个list，里面每一项是['T', 'h', 'e', ' ', 'a', 'd', 'm',...
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    print("字典的长度：")#字符级字典长度：95(autolabel)  131(dnrti)
    print(len(dico))#单词级字典长度：27347(autolabel) 7986(dnrti)

    return dico#存的是每个字符及其对应出现的次数


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    从字典创建映射（项目到 ID / ID 到项目）。
    项目按频率递减排序。
    """
    #print(dico.items())
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    #print(sorted_items)#频率高的字符排在前面

    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    return item_to_id, id_to_item
    #return item_to_id, id_to_item.keys()

#目前只用了上面两个函数
#####################################################################

def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    使用预训练值初始化网络参数。我们检查尺寸是否兼容。
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.创建一个 numpy 数组的共享对象。
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros 偏差用零初始化
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    用零替换字符串中的每个数字。
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    检查标签是否具有有效的 IOB 格式。
    IOB1 格式的标签被转换为 IOB2。
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iob_ranges(tags):
    """
    IOB -> Ranges
    """
    ranges = []
    def check_if_closing_range():
        if i == len(tags)-1 or tags[i+1].split('-')[0] == 'O':
            ranges.append((begin, i, type))
    
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'O':
            pass
        elif tag.split('-')[0] == 'B':
            begin = i
            type = tag.split('-')[1]
            check_if_closing_range()
        elif tag.split('-')[0] == 'I':
            check_if_closing_range()
    return ranges


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    用概率为 p 的未知词替换单例。
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    填充句子中单词的字符。
     输入：
         - 整数列表列表（单词列表，一个单词是 char 索引列表）
     输出：
         - 整数列表的填充列表
         - 整数列表的填充列表（其中字符反转）
         - 与每个单词的最后一个字符的索引对应的整数列表
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    获取句子数据并返回训练或评估函数的输入。
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters['word_dim']:
        input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if add_label:
        input.append(data['tags'])
    return input


def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, dictionary_tags):
    """
    Evaluate current model using CoNLL script.
    使用 CoNLL 脚本评估当前模型。
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input(data, parameters, False)
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            #分隔符控制 eval.1276213.output(单词跟标签之间)
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            #xxx=new_line
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")  #在eval.1276213.output换行符后面加的字符,空字符用来划分每一句话

    #print("33333")
    #print(xxx)
    #print(predictions[30])
    #print(predictions[31])
    #print(predictions[32])
    # Write predictions to disk and run CoNLL script externally
    ##将预测写入磁盘并在外部运行 CoNLL 脚本
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))  ########!!!!!!!改的这里  原来是\n
    
    #print("eval_script:"+eval_script)#./evaluation/conlleval

    #os.system('python %s < %s > %s' % (eval_script, output_path, scores_path))
    #os.system('perl %s < %s > %s' % (eval_script, output_path, scores_path))

    
    #os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

    os.system('%s -r -d "   " < %s > %s' % (eval_script, output_path, scores_path))
    #os.system('%s -r -d "   " < %s ' % (eval_script, scores_path))

    # CoNLL evaluation results
    print(scores_path)#./evaluation/temp/eval.1060503.scores
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    print("nihaoya")
    print(eval_lines)#[]
    #for line in eval_lines:
        #print("111111111")
        #print (line)

    # Remove temp files
    # os.remove(output_path)
    # os.remove(scores_path)

    # Confusion matrix with accuracy for each tag（每个标签准确度的混淆矩阵）
    print (("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(n_tags)] + ["Percent"])
    ))
    for i in range(n_tags):
        print (("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        ))

    # Global accuracy
    print("Global accuracy:")
    print ("%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))

    # F1 on all entities
    print("F1 on all entities:")
    print(float(eval_lines[1].strip().split()[-1]))
    return float(eval_lines[1].strip().split()[-1])
