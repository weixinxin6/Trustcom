import csv
import json

#_read_text函数是我新添加到该代码里的
def _read_text(input_file):
        lines = []
        with open(input_file, 'r',encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
        return lines

def dict_to_jsonfile(dictdata, dir_path):
    jsonfile_str = json.dumps(dictdata, ensure_ascii=False)
    fo = open(dir_path, "w", encoding='utf-8')
    fo.write(jsonfile_str)
    return 0


def jsonfile_to_dict(dir_path):
    fo = open(dir_path, "r", encoding='utf-8')
    dictdata = json.load(fo)
    return dictdata


def list_to_csv(header, list_data, dir_path, if_write_haeder):
    fo = open(dir_path, "w", encoding='utf-8')
    writer = csv.writer(fo)
    if if_write_haeder:
        writer.writerow(header)
    writer.writerows(list_data)
    fo.close()

def csv_to_list(dir_path):
    fo = open(dir_path, "r")
    csv_reader = csv.reader(fo)
    csv_header = next(csv_reader)
    csv_data = [row for row in csv_reader]

    return csv_header, csv_data

def str_to_txt(file_dir,text):
    try:
        fo = open(file_dir,"w")
        fo.write(text)
        fo.close()
    except Exception as e:
        print("File writing exception: ",e )


def txt_to_str(file_dir):
    try:
        fo = open(file_dir,"r")
        file_text = fo.read()
        fo.close()
    except Exception as e:
        print("File open exception: ",e )
        file_text = ""
    return file_text


def binary_to_file(binary, file_dir):
    try:
        fo = open(file_dir, "wb")
        fo.write(binary)
        fo.close()
    except Exception as e:
        print("File writing exception: ", e)

def listjson_to_list(file_dir):
    try:
        fo = open(file_dir, "r")
        json_list = []
        data_list = fo.readlines()
        for data in data_list:
            json_list.append(json.loads(data))
        fo.close()
        return json_list
    except Exception as e:
        print("File writing exception: ", e)
        return []

if __name__ == '__main__':
    root_path = "/home/youyizhe/PythonProject/DataCon2021/"
    all_data = listjson_to_list(root_path + "Data/20211012/web_data/data.json")
    print(len(all_data))
