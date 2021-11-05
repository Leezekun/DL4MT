import os
import re
import numpy as np
import json
import csv
import pickle

def clean_data():
    tsv_file = open("./data/opus.ha-en.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    f1 = open("./data/train_no_clean.en", "w")
    f2 = open("./data/train_no_clean.ha", "w")
    f3 = open("./data/batch_noise_data_only_return.json", "w")

    src_num = 0 
    trg_num = 0

    batch_size = 256

    batch_noise_data = []
    batch_contain_return = 0
    batch_contain_none = 0
    batch_contain_brackets = 0
    total_contain_return = 0
    total_contain_none = 0
    total_contain_brackets = 0

    for idx, row in enumerate(read_tsv):
        src = row[0].strip()
        trg = row[1].strip()
        
        if idx % batch_size == 0:
            batch_noise_data.append((batch_contain_return, batch_contain_none, batch_contain_brackets))
            batch_contain_return = 0
            batch_contain_none = 0
            batch_contain_brackets = 0
            
        if "\n" in src or "\n" in trg:
            # print(f"src:{src} \n tgt:{trg}")
            batch_contain_return += 1
            total_contain_return += 1
            continue
        if src is None or trg is None:
            # print(f"idx:{idx}, src:{src} \n tgt:{trg}")
            batch_contain_none += 1
            total_contain_none += 1
            continue
        # if re.search(r'.*:.*\) ', trg) or re.search(r'\(.*', trg):
        #     # print(f"idx:{idx}, src:{src} \n tgt:{trg}")
        #     trg = re.sub(r'.*:.*\) ', "", trg)
        #     trg = re.sub(r'\(.*', "", trg)
        #     batch_contain_brackets += 1
        #     total_contain_brackets += 1
        
        f1.write(src + "\n")
        src_num += 1
        f2.write(trg + "\n")
        trg_num += 1

    json.dump(batch_noise_data, f3)

    f1.close()
    f2.close()
    f3.close()

    print(f"Total num:{idx}, src num:{src_num}, trg num:{trg_num}")
    print(f"Total_contain_return:{total_contain_return}, total_contain_none:{total_contain_none}, total_contain_brackets:{total_contain_brackets}")


def create_data(args, logger):
    
    for mode in ["train", "dev"]:
        if mode == "train":
            if os.path.exists(args.train_file):
                return

            source_data = open(args.train_source_dataset, 'r')
            source_data = source_data.readlines()
            target_data = open(args.train_target_dataset, 'r')
            target_data = target_data.readlines()
            save_path = args.train_file

        elif mode == "dev":
            if os.path.exists(args.dev_file): 
                return

            source_data = open(args.dev_source_dataset, 'r')
            source_data = source_data.readlines()
            target_data = open(args.dev_target_dataset, 'r')
            target_data = target_data.readlines()
            save_path = args.dev_file
            

        assert len(source_data) == len(target_data)
        data_size = len(source_data)

        with open(args.source_vocab) as f:
            source_vocab = json.load(f)
        with open(args.target_vocab) as f:
            target_vocab = json.load(f)

        source_less_num = 0
        source_len = []
        split_source_data = []
        for data in source_data:
            tokens = data.split()
            split_source_data.append(tokens)
            source_len.append(len(tokens))
            if len(tokens) < args.max_src_len:
                source_less_num += 1
        
        target_less_num = 0
        target_len = []
        split_target_data = []
        for data in target_data:
            tokens = data.split()
            split_target_data.append(tokens)
            target_len.append(len(tokens))
            if len(tokens) < args.max_src_len:
                target_less_num += 1

        logger.info("Loading data")

        num_samples = len(split_source_data)
        len_mean = np.mean(source_len)
        len_median = np.median(source_len)
        len_max = np.max(source_len)
        source_less_ratio = source_less_num / num_samples

        logger.info("Mode:{}, Total source samples:{}, mean of sample len:{}, median of sample len:{}, max len:{}, less than {}:{}".format(mode, num_samples, len_mean, len_median, len_max, args.max_src_len, source_less_ratio))

        num_samples = len(split_target_data)
        len_mean = np.mean(target_len)
        len_median = np.median(target_len)
        len_max = np.max(target_len)
        target_less_ratio = target_less_num / num_samples

        logger.info("Mode:{}, Total target samples:{}, mean of sample len:{}, median of sample len:{}, max len:{}, less than {}:{}".format(mode, num_samples, len_mean, len_median, len_max, args.max_trg_len, target_less_ratio))

        EOS = '<EOS>'
        GO = '<GO>'
        UNK = '<UNK>'
        PAD = '<PAD>'

        source_eos_idx = source_vocab[EOS] 
        target_eos_idx = target_vocab[EOS]
        source_go_idx = source_vocab[GO] 
        target_go_idx = target_vocab[GO]
        source_unk_idx = source_vocab[UNK] 
        target_unk_idx = target_vocab[UNK]
        source_pad_idx = source_vocab[PAD] 
        target_pad_idx = target_vocab[PAD]

        processed_data = []

        for idx in range(data_size):
            # processing source data
            src_data = split_source_data[idx]
            src_tok_ids = np.ones([args.max_src_len], dtype=int) * source_pad_idx
            src_padding_ids = np.ones([args.max_src_len], dtype=int)
            for i, token in enumerate(src_data):
                if i == 0: 
                    src_tok_ids[i] = source_go_idx
                    src_padding_ids[i] = 0
                if token in source_vocab:
                    tok_id = source_vocab[token]
                else:
                    tok_id = source_unk_idx
                if i+1 < args.max_src_len:
                    src_tok_ids[i+1] = tok_id
                    src_padding_ids[i+1] = 0
            if i+2 < args.max_src_len:
                src_tok_ids[i+2] = source_eos_idx
                src_padding_ids[i+2] = 0

            # processing target data
            trg_data = split_target_data[idx]
            trg_tok_ids = np.ones([args.max_trg_len], dtype=int) * target_pad_idx
            for i, token in enumerate(trg_data):
                if i == 0: 
                    trg_tok_ids[i] = target_go_idx
                if token in target_vocab:
                    tok_id = target_vocab[token]
                else:
                    tok_id = target_unk_idx
                if i+1 < args.max_trg_len:
                    trg_tok_ids[i+1] = tok_id
            if i+2 < args.max_trg_len:
                trg_tok_ids[i+2] = target_eos_idx

            processed_data.append((src_tok_ids, src_padding_ids, trg_tok_ids))

        with open(save_path, "wb") as f:
            pickle.dump(processed_data, f)
        logger.info("finish preprocessing data for {}, store data in {}".format(mode, save_path))
        

if __name__ == '__main__':
    clean_data()