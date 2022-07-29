#task_name对应processor/bert_ner_seq.py里面的ner_processors:
  #支持dnrti（bio标注）、aptner(bioes标注)、autolabel（bio标注）、cti_report（bio标注）

#model_type：'bert'、'bert_bilstm_crf'、'bert_bilstm_atten_crf'、'bert_gru_crf'
#model_name_or_path：预训练模型的存储位置
#data_dir：数据集所在位置（CTIReports）
#output_dir：输出文件位置

#makeup：processor/utils_ner.py(get_entities_bio,get_entities_bioes,get_entity_bios),支持三种标注方式bio、bioes、bios

python method/bert_based.py \
  --task_name=aptner \
  --model_type=bert_bilstm_crf \
  --model_name_or_path=method/uncased_L-12_H-768_A-12/ \
  --data_dir=Data/APTNER/ \
  --output_dir=outputs/APTNER_bert/ \
  --markup=bioes \
  --language=en \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=7e-5 \
  --crf_learning_rate=1e-3 \
  --num_train_epochs=20.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --overwrite_output_dir \
  --seed=42 \
  --train_data_limit=-1
