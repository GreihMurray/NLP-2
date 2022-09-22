import csv
from tqdm import tqdm


def read_file():
    all_data = []

    with open("train.tsv", encoding="utf-8") as file:
        f = csv.reader(file, delimiter="\t")

        for line in tqdm(f, desc="Reading data..."):
            if line[0][0:2].strip() == 'N':
                line[0] = 'N'

            all_data.append((line[0], line[1]))

    return all_data


def read_file_to_sents():
    with open("train.tsv", encoding="utf-8") as file:
        f = csv.reader(file, delimiter="\t")

        cur_sent = []
        all_sents = []

        for line in tqdm(f, desc="Reading data..."):
            if line[0][0:2].strip() == 'N':
                line[0] = 'N'

            if line[0] == "<S>":
                if len(cur_sent) >= 1:
                    all_sents.append(cur_sent)
                cur_sent = []
                continue

            cur_sent.append((line[0], line[1]))

    return all_sents


def split_data(data):
    data_dict = {}

    i = 0

    for sent in tqdm(data, desc="Splitting data..."):
        cur_sent = []
        cur_tags = []
        for entry in sent:
            cur_sent.append(entry[0])
            cur_tags.append(entry[1])

        data_dict[i] = {'sent': cur_sent, 'tags': cur_tags}
        i += 1

    return data_dict