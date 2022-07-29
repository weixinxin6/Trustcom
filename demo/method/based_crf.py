import glob
import logging
import os
import json
import time

import gensim
import torch
import torch.nn as nn
import sys
#project_root_path = "/home/youyizhe/PythonProject/SecNER/"
project_root_path = "F:/A2_postgraduate/pt_model/demo_1/demo"
sys.path.append(project_root_path)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything, json_to_text, resume_origin_token
from tools.common import init_logger, logger

from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup,AdamW, BertTokenizer
from models.BiLSTM_Attention_CRF import BiLSTM_Atten_CRF
from models.BiLSTM_CRF import BiLSTM_CRF
from models.CRF_solo import CRF_solo
from processor.utils_ner import get_entities
from processor.example2feature import convert_examples_to_features
from processor.bert_ner_seq import ner_processors as processors
from processor.word2vec_ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.word2vec_argparse import get_argparse
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from torchsummary import summary #打印模型架构

'''global_step, tr_loss, best_eval_model, best_args, best_scheduler, best_F1_score, best_step 
= train(args, train_dataset, model,device)'''

def train(args, train_dataset, model, device_str=""):
    """ Train the model """

    #print("train_dataset:")
    #print(len(train_dataset))#5253

    args.train_batch_size = args.per_gpu_train_batch_size
    #print("args.train_batch_size:")
    #print(args.train_batch_size)#16   

    train_sampler = RandomSampler(train_dataset)#随机采样

    '''将数据分批的放入到训练网络中，
    批数量的大小也被称为batch_size,通过pytorch提供的dataloader方法，
    可以自动实现一个迭代器，每次返回一组batch_size个样本和标签来进行训练。'''
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    #print("train_dataloader:")
    #print(len(train_dataloader))#329  5253÷16≈328.3  把训练集分成329个样本

    #print("args.max_steps:")
    #print(args.max_steps)#-1

    #print("args.gradient_accumulation_steps:")
    #print(args.gradient_accumulation_steps)#1

    #print(args.num_train_epochs)#20.0

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:       
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        #print(t_total)#6580.0  迭代20次，总共有329×20个小样本
    
    #print(args.warmup_proportion)#0.1
    args.warmup_steps = int(t_total * args.warmup_proportion)
    #print(args.warmup_steps)#658 使用总样本的0.1进行热身

    # Prepare optimizer and schedule (linear warmup and decay)
    # 准备优化器和时间表（线性预热和衰减）
    no_decay = ["bias", "LayerNorm.weight"]

    '''list()函数是Python的内置函数。它可以将任何可迭代数据转换为列表类型,并返回转换后的列表。
    当参数为空时,list函数可以创建一个空列表。'''
    model_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    #print(args.learning_rate)#0.001
    #print(args.adam_epsilon)#1e-08
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))#5253
    logger.info("  Num Epochs = %d", args.num_train_epochs)#20
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)#16
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps,
                )#16
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)#1
    logger.info("  Total optimization steps = %d", t_total)#6580

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    best_F1_score = 0.0
    best_eval_model = None
    best_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    if args.save_steps == -1 and args.logging_steps == -1:
        args.logging_steps = len(train_dataloader)
        args.save_steps = len(train_dataloader)
    for epoch in range(int(args.num_train_epochs)):#20
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            #如果继续训练，跳过任何已训练的步骤
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            #summary(model,input_size=(16,37,100))
            batch = tuple(t.to(device_str) for t in batch)
            inputs = {"input_ids": batch[0], "input_len": batch[1], "labels": batch[2]}

            #print("inputs.get")
            #print(inputs.get("input_ids"))#与下面输出一样
            #print(inputs["input_ids"])

            #print(inputs["labels"])#形状：[16,37]

            logits, loss = model(inputs["input_ids"],inputs["labels"])

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    eval_result = evaluate(args, model, "", device_str, "dev")
                    if eval_result["f1"] > best_F1_score:
                        best_F1_score = eval_result["f1"]
                        best_eval_model = model
                        best_step = global_step
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(model,
                                                             "module") else model)
                    torch.save(model_to_save, os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    logger.info("\n")
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

    # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(best_step))
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # model_to_save = (
    #     best_eval_model.module if hasattr(model, "module") else model
    # )  # Take care of distributed/parallel training
    # torch.save(model_to_save, os.path.join(output_dir, "pytorch_model.bin"))
    # logger.info("Saving model binary file to %s", output_dir)
    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
    # logger.info("Saving model checkpoint to %s", output_dir)
    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    # logger.info("Saving optimizer and scheduler states to %s", output_dir)

    # return global_step, tr_loss / global_step
    return global_step, tr_loss / global_step, best_eval_model, args, scheduler, best_F1_score, best_step


def evaluate(args, model, prefix="", device_str="", data_type='dev'):

    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # if args.task_name.find("general") >= 0:
    #     w2v_model = gensim.models.KeyedVectors.load_word2vec_format(args.embed_path, binary=True)
    #     w2v_embedding = w2v_model.vectors
    # else:
    #     w2v_model = Word2Vec.load(args.embed_path)
    #     w2v_embedding = w2v_model.wv.vectors

    eval_dataset = load_and_cache_examples(args, args.task_name, data_type=data_type)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(device_str) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "input_lens": batch[1], "labels": batch[2]}
            logits, loss = model(inputs["input_ids"], inputs["labels"])
            tags = model.crf.decode(logits)
        eval_loss += loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[1].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {"ner-eval-" + key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results


def load_and_cache_examples(args, task, data_type='train', limit=-1):
    processor = processors[task]()
    # Load data features from cache or dataset file
         #cached_crf-   train_    uncased_L-12_H-768_A-12_   128_   dnrti
    cached_features_file = os.path.join(args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_type.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task)))
    #wei#if task.find("general") >= 0:
        #wei#key_index_dict = w2v_model.key_to_index
    #wei#else:
        #wei#key_index_dict = w2v_model.wv.key_to_index
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        #print("examples")
        #print(examples[33])#单词和标签都有
        #print(label_list)
        features = convert_examples_to_features(examples, label_list, max_seq_length=args.train_max_seq_length,
                pad_label_ids=0)

   # Convert to Tensors and build dataset
    if limit != -1:
        features = features[:limit]#切片设置数据集大小

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_lens, all_label_ids)
    return dataset


def main():
    args = get_argparse().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        device = "cuda:" + str(args.device_num)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H：%M：%S", time.localtime())
    ###wei### embedding_name = args.embed_path.split("/")[-1].split("_")[0]
    embedding_name="embedding"

    init_logger(log_file=args.output_dir + f'/{args.model_type}-{embedding_name}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    #print("label_list:")
    #print(label_list)#['O', 'B-HackOrg', 'I-HackOrg', ...
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    #print(args.id2label)#{0: 'O', 1: 'B-HackOrg', ...
    #print(args.label2id)#{'O': 0, 'B-HackOrg': 1, 'I-HackOrg': 2, 
    num_labels = len(label_list)

    # Load pretrained model and tokenizer

    args.model_type = args.model_type.lower()
    # 判断模型类型并实例化模型
    #wei#if args.task_name.find("general") >= 0:
        #wei#w2v_model = gensim.models.KeyedVectors.load_word2vec_format(args.embed_path, binary=True)
        #wei#w2v_embedding = w2v_model.vectors
    #wei#else:
        #wei#w2v_model = Word2Vec.load(args.embed_path)
        #wei#w2v_embedding = w2v_model.wv.vectors

    #wei#vocab_size, embedding_dim = w2v_embedding.shape
    #wei#embedding_ = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    #wei#embedding_.weight.data.copy_(torch.from_numpy(w2v_embedding))
    #wei#embedding_.weight.requires_grad = False
    
    if args.model_type.find("bilstm_crf") >= 0:
        print("bilstm_crf")
        #print(args.lstm_hidden_dim)#隐藏层维度 100
        #print(args.lstm_layers)#LSTM层数 1
        #print(num_labels) #标签数量 29
        #第一个参数是嵌入维度，这里指定100

        #dnrti:num_embeddings=7986   autolabel:27347
        embedding_ = nn.Embedding(num_embeddings=7986, embedding_dim=100)
        model = BiLSTM_CRF(args.lstm_hidden_dim, args.lstm_layers, num_labels,embedding_)
    elif args.model_type.find("bilstm_atten_crf") >= 0:
        print("bilstm_atten_crf")
        model = BiLSTM_Atten_CRF(embedding_, args.lstm_hidden_dim, args.lstm_layers, num_labels)
    else:
        print("crf_solo")
        model = CRF_solo(embedding_, num_labels)
    model.to(device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        #print("args.train_data_limit")
        #print(args.train_data_limit) # -1
        train_dataset = load_and_cache_examples(args, args.task_name, data_type='train',        
                                                            limit=args.train_data_limit)
        #print("train函数的参数:")
        #print(model)
        #print(device)#cpu
        global_step, tr_loss, best_eval_model, best_args, best_scheduler, best_F1_score, best_step = train(args, train_dataset, model,
                                                                                device)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = (
            best_eval_model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        torch.save(model_to_save, os.path.join(args.output_dir, "pytorch_model.bin"))
        logger.info("Saving model binary file to %s", args.output_dir)
        torch.save(best_args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(best_scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", args.output_dir)

    # Evaluation
    results = {}
    if args.do_eval:
        model = torch.load(os.path.join(args.output_dir, "pytorch_model.bin"))
        model.to(device)
        prefix = ''
        result = evaluate(args, model, prefix, device, "test")
        results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))


if __name__ == "__main__":
    main()
