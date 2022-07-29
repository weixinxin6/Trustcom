from processor.bert_ner_seq import ner_processors


dataset_name = "dnrti"
entity_type_list = []
task_processor = ner_processors[dataset_name]()
label_list = task_processor.get_labels()
print(list(set(
    [
        lab_.split("-")[1] for lab_ in label_list if lab_.startswith("B-") or lab_.startswith("I-")]
)))