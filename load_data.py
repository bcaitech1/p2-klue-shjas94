import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pororo import Pororo
# Dataset 구성.


class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.


def delete_duplicate(df):
    return df.drop_duplicates([1])


def delete_duplicate2(df):
    return df.drop_duplicates(['sentence'])


def preprocessing_dataset(dataset, label_type, train=True):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {'sentence': dataset[1], 'entity_01': dataset[2], 'entity_01_spos': dataset[3], 'entity_01_epos': dataset[4],
         'entity_02': dataset[5], 'entity_02_spos': dataset[6], 'entity_02_epos': dataset[7], 'label': label, })

    if train:
        # 추가 데이터
        # df_born_city = pd.read_csv(
        #     "/opt/ml/input/data/train/19_bornIn_city.tsv", delimiter='\t', header=0)
        # df_born_ctry = pd.read_csv(
        #     "/opt/ml/input/data/train/26_bornIn_country.tsv", delimiter='\t', header=0)
        # df_death_city = pd.read_csv(
        #     "/opt/ml/input/data/train/37_dienIn_city.tsv", delimiter='\t', header=0)
        # df_death_ctry = pd.read_csv(
        #     "/opt/ml/input/data/train/40_dienIn_country.tsv", delimiter='\t', header=0)

        # df_born_city = df_born_city.iloc[:10]
        # # df_born_city = delete_duplicate2(df_born_city).iloc[:100]
        # df_born_ctry = df_born_ctry.iloc[:10]
        # # df_born_ctry = delete_duplicate2(df_born_ctry).iloc[:100]
        # df_death_city = df_death_city.iloc[:10]
        # # df_death_city = delete_duplicate2(df_death_city).iloc[:100]
        # df_death_ctry = df_death_ctry.iloc[:10]
        # # df_death_ctry = delete_duplicate2(df_death_ctry).iloc[:100]
        # out_dataset = pd.concat([out_dataset, df_born_city, df_born_ctry,
        #                         df_death_city, df_death_ctry], axis=0, ignore_index=True)
        # out_dataset = delete_duplicate2(out_dataset)
        additional = pd.read_csv(
            "/opt/ml/input/data/train/all_csv.tsv", delimiter='\t', header=None)
        additional = additional.drop_duplicates(subset=[1])
        additional = additional.sample(9000, random_state=42)
        add_label = []
        for i in additional[8]:
            if i == 'blind':
                add_label.append(100)
            else:
                add_label.append(label_type[i])

        # additional_set = pd.DataFrame(
        #     {'sentence': additional[1], 'entity_01': additional[2], 'entity_02': additional[5], 'label': add_label, })
        additional_set = pd.DataFrame(
            {'sentence': additional[1], 'entity_01': additional[2], 'entity_01_spos': additional[3], 'entity_01_epos': additional[4],
             'entity_02': additional[5], 'entity_02_spos': additional[6], 'entity_02_epos': additional[7], 'label': label, })
        train, val = train_test_split(
            out_dataset, test_size=0.2, shuffle=True, random_state=42)

        train = pd.concat([train, additional_set], axis=0)
    ##########
        return train, val
    else:
        return out_dataset

# tsv 파일을 불러옵니다.


def load_data(dataset_dir, train=True):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # dataset = dataset.drop_duplicates(subset=[1, 2, 5])
    # dataset = delete_duplicate(dataset)
    ### 추가 데이터셋 투입(라벨별 100개씩만) ###

    ##########################################
    # preprecessing dataset
    if train:
        # additional_dataset = pd.read_csv(
        #     "/opt/ml/input/data/train/all_csv.tsv", delimiter='\t', header=None)
        # # labels = additional_datast[8].unique()
        # # for label in labels:
        # #     temp_df = additional_datast[additional_datast[8] == label]
        # #     dataset = pd.concat([dataset, temp_df.iloc[:100]],
        # #                         axis=0, ignore_index=True)
        # additional_dataset = additional_dataset.sample(4000, random_state=42)
        # dataset = pd.concat([dataset, additional_dataset],
        #                     axis=0, ignore_index=True)
        train, val = preprocessing_dataset(dataset, label_type)
        return train, val
    else:
        dataset = preprocessing_dataset(dataset, label_type, train=False)
        return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.


def tokenized_dataset(dataset, tokenizer):
    # concat_entity = []
    # for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    #     temp = ''
    #     # temp = e01 + '[SEP]' + e02

    #     # xlm Roberta
    #     temp = e01 + '</s>' + e02
    #     concat_entity.append(temp)
    ner = Pororo(task="ner", lang="ko")

    fixed_sents = []

    for sent, ent01, start1, end1, ent02, start2, end2 in zip(dataset['sentence'], dataset['entity_01'], dataset['entity_01_spos'], dataset['entity_01_epos'], dataset['entity_02'], dataset['entity_02_spos'], dataset['entity_02_epos']):
        ner_01 = ' | ' + ner(ent01)[0][1].lower() + ' | '
        ner_02 = ' ^ ' + ner(ent02)[0][1].lower() + ' ^ '

        if start1 < start2:
            sent = sent[:start1] + '@' + ner_01 + sent[start1:end1 + 1] + ' @ ' + sent[end1 +
                                                                                       1: start2] + '#' + ner_02 + sent[start2: end2 + 1] + ' # ' + sent[end2 + 1:]
        else:
            sent = sent[:start2] + '#' + ner_01 + sent[start2:end2 + 1] + ' # ' + sent[end2 +
                                                                                       1: start1] + '@' + ner_02 + sent[start1: end1 + 1] + ' @ ' + sent[end1 + 1:]

        fixed_sents.append(sent)

    concat_entity = []

    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '</s>' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=120,
        add_special_tokens=True,
    )
    return tokenized_sentences
