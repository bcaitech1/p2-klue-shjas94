import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from transformers import ElectraForSequenceClassification, ElectraConfig, ElectraTokenizer
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig, XLMRobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
from load_data import *
import wandb
import argparse
import numpy as np
import random
from sklearn.model_selection import train_test_split

# 평가를 위한 metrics function.


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    # wandb
    wandb.init(project='P_Stage2', entity='shjas94',
               tags=args.tag_list, group=args.group, name=args.run_name)
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = 'true'
    # load model and tokenizer
    MODEL_NAME = "bert-base-multilingual-cased"
    seed_everything(args.seed)
    # electra
    MODEL_NAME2 = "monologg/koelectra-base-v3-discriminator"
    MODEL_NAME3 = "monologg/koelectra-small-v3-discriminator"
    MODEL_NAME4 = "xlm-roberta-large"
    # baseline tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # electra tokenizer
    # tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME2)

    # xlm-Roberta tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME4)

    # load dataset
    train_dataset, val_dataset = load_data(
        "/opt/ml/input/data/train/train.tsv")
    #dev_dataset = load_data("./dataset/train/dev.tsv")
    train_label = train_dataset['label'].values
    val_label = val_dataset['label'].values
    #dev_label = dev_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_val = tokenized_dataset(val_dataset, tokenizer)
    #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_val_dataset = RE_Dataset(tokenized_val, val_label)
    #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    # bert_config = BertConfig.from_pretrained(MODEL_NAME)
    # bert_config.num_labels = 42
    # model = BertForSequenceClassification(bert_config)
    # model.parameters
    # model.to(device)

    # setting electra
    # electra_config = ElectraConfig.from_pretrained(MODEL_NAME2)
    # electra_config.num_labels = 42
    # model = ElectraForSequenceClassification.from_pretrained(
    #     MODEL_NAME2, config=electra_config)
    # # model.classifier.dropout = torch.nn.Dropout(p=0.7, inplace=False)
    # model.to(device)

    # # setting xlm RoBerta
    xlm_roberta_config = XLMRobertaConfig.from_pretrained(MODEL_NAME4)
    xlm_roberta_config.num_labels = 42
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME4, config=xlm_roberta_config)
    # model.parameters
    # model.classifier.dropout = torch.nn.Dropout(p=0.7, inplace=False)
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=args.output_dir,          # output directory
        # number of total save model.
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,                 # model saving step.
        # total number of training epochs
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,               # learning_rate
        # batch size per device during training
        per_device_train_batch_size=args.per_device_train_batch_size,
        # per_device_eval_batch_size=16,   # batch size for evaluation
        # number of warmup steps for learning rate scheduler
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,               # strength of weight decay
        logging_dir=args.logging_dir,            # directory for storing logs
        logging_steps=args.logging_steps,              # log saving step.

        run_name=args.run_name,
        evaluation_strategy='steps',
        # load_best_model_at_end=True
        # evaluation_strategy='steps', # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=250,            # evaluation step.
    )
    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()


def main(args):
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="P Stage2 KLUE Competition")
    parser.add_argument('--seed', required=False, default=42)
    parser.add_argument('--output_dir', required=False, default='./results')
    parser.add_argument('--save_total_limit',
                        required=False, type=int, default=3)
    parser.add_argument('--save_steps', required=False, type=int, default=250)
    parser.add_argument('--num_train_epochs',
                        required=False, type=int, default=5)
    parser.add_argument('--learning_rate', required=False,
                        type=float, default=5e-5)
    parser.add_argument('--per_device_train_batch_size',
                        required=False, type=int, default=16)
    parser.add_argument('--warmup_steps', required=False,
                        type=int, default=500)
    parser.add_argument('--weight_decay', required=False,
                        type=float, default=0.01)
    parser.add_argument('--logging_dir', required=False, default='./logs')
    parser.add_argument('--logging_steps', required=False,
                        type=int, default=100)
    parser.add_argument('--report_to', required=False, default='wandb')
    parser.add_argument('--group', required=True)
    parser.add_argument('--model', required=True)

    parser.add_argument('--run_name', required=True)
    parser.add_argument('--tag', required=True,
                        action='append', dest='tag_list')

    args = parser.parse_args()
    main(args)
