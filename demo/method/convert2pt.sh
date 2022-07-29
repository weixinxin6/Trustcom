#将tensorflow版本的bert预训练模型转换为pytorch版本

export BERT_BASE_DIR=method/xlnet_cased_L-12_H-768_A-12

transformers-cli convert --model_type xlnet \
  --tf_checkpoint $BERT_BASE_DIR/xlnet_model.ckpt \
  --config $BERT_BASE_DIR/xlnet_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/