#task_name对应processor/bert_ner_seq.py里面的ner_processors:
  #支持dnrti（bio标注）、aptner(bioes标注)、autolabel（bio标注）、cti_report（bio标注）

#model_type = bilstm_crf、gru_crf

#data_dir：数据集所在位置（CTIReports）


python method/word2vec_based_crf.py \
  --task_name=autolabel \
  --model_type=gru_crf \
  --data_dir=Data/AutoLabel/ \
  --output_dir=outputs/AutoLabel_word2vec/ \
  --embed_path=embeddings/AutoLabel_word2vec.model \
  --markup=bio \
  --learning_rate=1e-3 \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --num_train_epochs=20.0 \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --logging_steps=-1 \
  --save_steps=-1 \
  --device_num=1 \
  --train_data_limit=-1