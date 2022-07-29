#运行crf_solo、bilstm_crf或bilstm_atten_crf模型

python method/based_crf.py \
  --task_name=DNRTI \
  --model_type=bilstm_crf \
  --data_dir=Data/DNRTI/ \
  --output_dir=outputs/DNRTI_crf/ \
  --embed_path="" \
  --markup=bioes \
  --do_train \
  --do_eval \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=20.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --overwrite_output_dir \
  --seed=42 \
  --train_data_limit=-1