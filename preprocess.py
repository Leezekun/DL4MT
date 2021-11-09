import os
import re
import numpy as np
import json
import csv
import pickle

def remove_html(htmlstr):
    re_cdata = re.compile('//<![CDATA[[^>]*//]]>',re.I)#匹配CDATA
    re_script = re.compile('<s*script[^>]*>[^<]*<s*/s*scripts*>',re.I)#Script
    re_style=re.compile('<s*style[^>]*>[^<]*<s*/s*styles*>',re.I)#style
    re_h=re.compile('</?w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    re_br2 = re.compile("<[^>]+?>")
    re_br3 = re.compile("\[[^>]+?\]")

    s = htmlstr
    # s = re_cdata.sub('',htmlstr)#去掉CDATA
    # s = re_script.sub('',s) #去掉SCRIPT
    # s = re_style.sub('',s)#去掉style 
    s = re_br2.sub('',s)#将br转换为换行
    s = re_br3.sub('',s)#将br转换为换行

    s = re_h.sub('',s) #去掉HTML 标签
    s = re_comment.sub('',s)#去掉HTML注释
    return s

def remove_time(str_sentence):
    str_sentence = re.sub('^[^\)]*:[^\)]*\)', "", str_sentence)
    str_sentence = re.sub('\([^\)]*$', "", str_sentence)
    return str_sentence

def remove_newline(str_sentence):
    str_sentence = re.sub('\s+'," ",str_sentence)
    return str_sentence

def process_string(str_sentence):
    # str_sentence = remove_html(str_sentence)
    str_sentence = remove_time(str_sentence)
    str_sentence = remove_newline(str_sentence)
    str_sentence = str_sentence.strip()
    return str_sentence

def clean_data():
    tsv_file = open("./data/opus.ha-en.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    f1 = open("./data/train_clean.en", "w")
    f2 = open("./data/train_clean.ha", "w")
    f3 = open("./data/batch_noise_data.json", "w")

    src_num = 0 
    trg_num = 0

    filtered = 0
    
    # batch_size = 256
    # batch_noise_data = []
    # batch_contain_return = 0
    # batch_contain_none = 0
    # batch_contain_brackets = 0
    # total_contain_return = 0
    # total_contain_none = 0
    # total_contain_brackets = 0

    for idx, row in enumerate(read_tsv):
        source = row[0].strip()
        target = row[1].strip()
        
        # if idx % batch_size == 0:
        #     batch_noise_data.append((batch_contain_return, batch_contain_none, batch_contain_brackets))
        #     batch_contain_return = 0
        #     batch_contain_none = 0
        #     batch_contain_brackets = 0
            
        # if "\n" in src or "\n" in trg:
        #     # print(f"src:{src} \n tgt:{trg}")
        #     batch_contain_return += 1
        #     total_contain_return += 1
        #     continue
        # if src is None or trg is None:
        #     # print(f"idx:{idx}, src:{src} \n tgt:{trg}")
        #     batch_contain_none += 1
        #     total_contain_none += 1
        #     continue
        # if re.search(r'.*:.*\) ', trg) or re.search(r'\(.*', trg):
        #     # print(f"idx:{idx}, src:{src} \n tgt:{trg}")
        #     trg = re.sub(r'.*:.*\) ', "", trg)
        #     trg = re.sub(r'\(.*', "", trg)
        #     batch_contain_brackets += 1
        #     total_contain_brackets += 1

        src = source.lower()
        trg = target.lower()
        src = process_string(src)
        trg = process_string(trg)

        if len(src) != 0 and len(trg) != 0:
            f1.write(src + "\n")
            src_num += 1
            f2.write(trg + "\n")
            trg_num += 1
        else:
            filtered += 1
            print(f"idx:{idx}")
            print(f"Original source:{source}, target:{target}")
            print(f"After processed, source:{src}, target:{trg}")
            print("\n")

    # json.dump(batch_noise_data, f3)

    f1.close()
    f2.close()
    f3.close()

    print(f"Total num:{idx}, filtered num:{filtered}, src num:{src_num}, trg num:{trg_num}")
    # print(f"Total_contain_return:{total_contain_return}, total_contain_none:{total_contain_none}, total_contain_brackets:{total_contain_brackets}")

    f1 = open("./data/dev.en", "w")
    f2 = open("./data/dev.ha", "w")
    with open("./data/dev/xml/newsdev2021.en-ha.en", "r") as f:
        data = f.readlines()
        for d in data:
            f1.write(d.lower())
    f1.close()
    with open("./data/dev/xml/newsdev2021.en-ha.ha", "r") as f:
        data = f.readlines()
        for d in data:
            f2.write(d.lower())
    f2.close()
    

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
            trg_padding_ids = np.ones([args.max_trg_len], dtype=int)
            trg_tok_ids = np.ones([args.max_trg_len], dtype=int) * target_pad_idx
            for i, token in enumerate(trg_data):
                if i == 0: 
                    trg_tok_ids[i] = target_go_idx
                    trg_padding_ids[i] = 0
                if token in target_vocab:
                    tok_id = target_vocab[token]
                else:
                    tok_id = target_unk_idx
                if i+1 < args.max_trg_len:
                    trg_tok_ids[i+1] = tok_id
                    trg_padding_ids[i+1] = 0
            if i+2 < args.max_trg_len:
                trg_tok_ids[i+2] = target_eos_idx
                trg_padding_ids[i+2] = 0
            
            processed_data.append((src_tok_ids, src_padding_ids, trg_tok_ids, trg_padding_ids))


        with open(save_path, "wb") as f:
            pickle.dump(processed_data, f)
        logger.info("finish preprocessing data for {}, store data in {}".format(mode, save_path))


if __name__ == '__main__':
    clean_data()