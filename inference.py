from transformers import AutoTokenizer, BertForSequenceClassification, ElectraForSequenceClassification, Trainer, TrainingArguments, BertConfig, ElectraConfig, ElectraTokenizer
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig, XLMRobertaTokenizer

from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)

    return np.array(output_pred).flatten()


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir, train=False)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label


def main(args):
    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    # TOK_NAME = "bert-base-multilingual-cased"
    TOK_NAME2 = "monologg/koelectra-base-v3-discriminator"
    TOK_NAME3 = "monologg/koelectra-small-v3-discriminator"
    TOK_NAME4 = "xlm-roberta-large"

    # tokenizer = ElectraTokenizer.from_pretrained(TOK_NAME2)
    tokenizer = XLMRobertaTokenizer.from_pretrained(TOK_NAME4)
    # load my model
    MODEL_NAME = args.model_dir  # model dir.
    model = XLMRobertaForSequenceClassification.from_pretrained(args.model_dir)
    # model = ElectraForSequenceClassification.from_pretrained(args.model_dir)
    model.parameters
    model.to(device)

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    # predict answer
    pred_answer = inference(model, test_dataset, device)
    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('./prediction/submission30.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', default="./results/checkpoint-4000")
    args = parser.parse_args()
    print(args)
    main(args)
