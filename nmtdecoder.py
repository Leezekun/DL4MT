import enum
import torch
from torch.utils.data import Dataset, DataLoader
from sacrebleu.metrics import BLEU

import numpy as np
import json
import math
import time
import os
import argparse

from preprocess import *
from hybridnmt import *
from beam import beam_search_decoding, batch_beam_search_decoding


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='7', type=str, required=False, help='')
    parser.add_argument('--no_cuda', action='store_true', help='')
    parser.add_argument('--log_path', default='./log/decode.log', type=str, required=False, help='')
    parser.add_argument('--train_file', default='./data/train.pkl', type=str, required=False, help='')
    parser.add_argument('--dev_file', default='./data/dev.pkl', type=str, required=False, help='')
    parser.add_argument('--output_file', default='./data/translation.txt', type=str, required=False, help='')
    parser.add_argument('--batch', default=True, type=bool, required=False, help='')
    parser.add_argument('--num_workers', default=8, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=256, type=int, required=False, help='')

    parser.add_argument('--max_src_len', default=128, type=int, required=False, help='')
    parser.add_argument('--max_trg_len', default=128, type=int, required=False, help='')

    parser.add_argument('--best_model_save_path', default="./model/best_whole_model/hybridnmt.pt", type=str, required=False, help='')
    parser.add_argument('--codes_file', default="./data/codes.en-ha", type=str, required=False, help='')
    parser.add_argument('--source_vocab_file', default="./data/vocab.en", type=str, required=False, help='')
    parser.add_argument('--target_vocab_file', default="./data/vocab.ha", type=str, required=False, help='')

    parser.add_argument('-i', default='./data/source.txt', type=str, required=False, help='')
    parser.add_argument('-o', default='./data/target.txt', type=str, required=False, help='')
    parser.add_argument('-eval', default='', type=str, required=False, help='')
    parser.add_argument('-dict', default='./data/joint_vocab.json', type=str, required=False, help='')

    parser.add_argument('--beam_width', type=int, default=10)
    parser.add_argument('--n_best', type=int, default=5)
    parser.add_argument('--max_dec_steps', type=int, default=1000)

    args = parser.parse_args()
    return args


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        src_ids, src_mask = self.data[index]        
            
        return {"src_ids": src_ids,
                "src_mask": src_mask
                }

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
    src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.bool)

    
    return {"src_ids": src_ids,
            "src_mask": src_mask
            }


def main():

    args = set_args()
    logger = create_logger(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    with open(args.dict) as f:
        vocab = json.load(f)
    f.close()
    print(f"Vocab size:{len(vocab)}")

    str2idx = vocab
    idx2token = {}
    for k, v in str2idx.items():
        idx2token[v] = k

    EOS = '<EOS>'
    GO = '<GO>'
    UNK = '<UNK>'
    PAD = '<PAD>'

    EOS_IDX = vocab[EOS] 
    GO_IDX = vocab[GO]
    UNK_IDX = vocab[UNK] 
    PAD_IDX = vocab[PAD]
    
    bleu = BLEU()
    
    model = torch.load(args.best_model_save_path)
    model.eval()
    
    sys = []
    ref = []

    input_file = args.i
    output_file = args.o

    if input_file[-4:] == '.xml':
        cmd = f'wmt-unwrap -o test_data < {input_file}'
        os.system(cmd)
    source = "test_data.en"
    
    # process the input data
    tokenize_cmd = f"subword-nmt apply-bpe -c {args.codes_file} --vocabulary {args.source_vocab_file} --vocabulary-threshold 50 < {args.i} > {args.i}.BPE"
    os.system(tokenize_cmd)

    processed_data = []
    with open(f"{args.i}.BPE", "r") as f:
        src = f.readline()
        src = re.sub('\s+'," ", src)
        src = src.split()
        src_tok_ids = np.ones([args.max_src_len], dtype=int) * PAD_IDX
        src_padding_ids = np.ones([args.max_src_len], dtype=int)
        for i, token in enumerate(src):
            if i == 0: 
                src_tok_ids[i] = GO_IDX
                src_padding_ids[i] = 0
            if token in vocab:
                tok_id = vocab[token]
            else:
                tok_id = UNK_IDX
            if i+1 < args.max_src_len:
                src_tok_ids[i+1] = tok_id
                src_padding_ids[i+1] = 0
        if i+2 < args.max_src_len:
            src_tok_ids[i+2] = GO_IDX
            src_padding_ids[i+2] = 0
        processed_data.append((src_tok_ids, src_padding_ids))

    my_dataset = MyDataset(processed_data)
    iterator = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)


    with torch.no_grad():
        for batch_id, batch in enumerate(iterator):
            src = batch['src_ids'].to(args.device)
            src_mask = batch['src_mask'].to(args.device)
            trg = batch['trg_ids'].to(args.device)
            trg_mask = batch['trg_mask'].to(args.device)
    
            ground_truth = trg 

            src = torch.transpose(src, 1, 0) # src = [src_len, batch_size]
            trg = torch.transpose(trg, 1, 0) # trg = [trg_len, batch_size]

            hidden1 = hidden2 = cell1 = cell2 = torch.zeros(args.batch_size, model.decoder.dec_hid_dim).to(args.device)
            input_feed = torch.zeros(args.batch_size, model.decoder.dec_out_dim).to(args.device)
            src = model.embedding(src) * math.sqrt(model.emb_dim) #src = [src_len, batch_size, emb_dim]
            enc_outs = model.encoder(src, src_mask)

            if not args.batch:
                start_time = time.time()
                decoded_seqs = beam_search_decoding(decoder=model.decoder,
                                                    enc_outs=enc_outs,
                                                    enc_masks=src_mask,
                                                    dec_hiddens1=hidden1,
                                                    dec_cells1=cell1,
                                                    dec_hiddens2=hidden2,
                                                    dec_cells2=cell2,
                                                    dec_input_feeds=input_feed,
                                                    embeddings=model.embedding,
                                                    beam_width=args.beam_width,
                                                    n_best=args.n_best,
                                                    sos_token=GO_IDX,
                                                    eos_token=EOS_IDX,
                                                    max_dec_steps=args.max_dec_steps,
                                                    device=args.device)
                end_time = time.time()
                print(f'Batch:{batch_id}, for loop beam search time: {end_time-start_time:.3f}')
            else:
                start_time = time.time()
                decoded_seqs = batch_beam_search_decoding(decoder=model.decoder,
                                                    enc_outs=enc_outs,
                                                    enc_masks=src_mask,
                                                    dec_hiddens1=hidden1,
                                                    dec_cells1=cell1,
                                                    dec_hiddens2=hidden2,
                                                    dec_cells2=cell2,
                                                    dec_input_feeds=input_feed,
                                                    embeddings=model.embedding,
                                                    beam_width=args.beam_width,
                                                    n_best=args.n_best,
                                                    sos_token=GO_IDX,
                                                    eos_token=EOS_IDX,
                                                    max_dec_steps=args.max_dec_steps,
                                                    device=args.device)
                end_time = time.time()
                print(f'Batch:{batch_id}, for batch beam search time: {end_time-start_time:.3f}')

            
            for bid, decoded_seq in enumerate(decoded_seqs):
                tokens = decoded_seq[0] # only use the top one
                sentence = []
                word = ""
                for tid, token in enumerate(tokens[1:]):
                    if token == EOS_IDX:
                        break
                    token = idx2token[token]
                    if token[-2:] == '@@': 
                        word += token[:-2]
                    else:
                        word += token
                        sentence.append(word)
                        word = ""
                sentence = " ".join(sentence)
                sys.append(sentence)

            trg = batch['trg_ids'].tolist()
            for bid, target in enumerate(trg):
                tokens = target # only use the top one
                sentence = []
                word = ""
                for tid, token in enumerate(tokens[1:]):
                    if token == EOS_IDX:
                        break
                    token = idx2token[token]
                    if token[-2:] == '@@': 
                        word += token[:-2]
                    else:
                        word += token
                        sentence.append(word)
                        word = ""
                sentence = " ".join(sentence)
                ref.append([sentence])
    
    


if __name__ == '__main__':
    main()