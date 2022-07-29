import gensim
from gensim.models import Word2Vec
import sys

project_root_path = "F:/A2_postgraduate/pt_model/demo_1/demo"
#project_root_path = "/root/demo_1/demo"
sys.path.append(project_root_path)

from util import file_util

def data_prepare(dataset_name,data_path):
    if(dataset_name=="AutoLabel"):
        train_json = file_util.jsonfile_to_dict(data_path + "train_data.json")
        dev_json = file_util.jsonfile_to_dict(data_path + "dev_data.json")
        test_json = file_util.jsonfile_to_dict(data_path + "test_data.json")
        all_list = train_json + dev_json + test_json
        #print("all_list")
        #print(all_list)
        #print(type(all_list))

        sentences = []
        for temp_data in all_list:
            #print(temp_data)#句子和标签
            #print(temp_data[0])#只有句子
            #break
            sentences.append(gensim.utils.simple_preprocess(temp_data[0]))
        print(sentences[0])#['the', 'authorize', 'net', 'echeck', '

    elif(dataset_name=="DNRTI" or dataset_name=="APTNER" or dataset_name=="CTIReports"):
        dev_txt = file_util._read_text(data_path+"dev.txt")
        train_txt = file_util._read_text(data_path+"train.txt")
        test_txt = file_util._read_text(data_path+"test.txt")
        all_list = train_txt+dev_txt+test_txt

        #print("dev_txt")
        #print(dev_txt)#{'words': ['The', 'GCMAN', ... 'out', '.'], 'labels': ['O', 'B-HackOrg', ']}
        #print(dev_txt[0]['words'])#['We', 'believe', 'that', 'these', 'to', 'healthcare', '.']
        #print(type(all_list))

        sentences = []
        for temp_data in all_list:
            #print(temp_data['words'])#只有句子
            sentences.append(temp_data['words'])
        print(sentences[0])
    
    return sentences

def word2vec_model_generate(text_list, save_path):

    model = Word2Vec(sentences=text_list, vector_size=200, window=5, min_count=1, workers=4)
    model.build_vocab(text_list)
    model.train(text_list, total_examples=len(text_list), epochs=20)
    model.save(save_path)

if __name__ == '__main__':

    # 需要训练词嵌入的数据集
    dataset_name = "CTIReports" #AutoLabel、APTNER、DNRTI、CTIReports

    model_save_path = "embeddings/" + dataset_name + "_word2vec.model"
    text_list = data_prepare(dataset_name,"Data/" + dataset_name + "/")
    #print(text_list)

    word2vec_model_generate(text_list, model_save_path)

    # 模型加载，不需要用到
    # model = Word2Vec.load(dataset_name + "_word2vec.model")





