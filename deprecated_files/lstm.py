import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from torch.utils.data import Dataset, DataLoader

from preprocess import *
from plot_training_curve import plot_curve

import numpy as np
import json
import pickle
import random
import math
import time
import os
import argparse
import logging


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='4', type=str, required=False, help='')
    parser.add_argument('--no_cuda', action='store_true', help='')
    parser.add_argument('--log_path', default='./log/train.log', type=str, required=False, help='')

    parser.add_argument('--train_source_dataset', default='./data/train.BPE.en', type=str, required=False, help='')
    parser.add_argument('--train_target_dataset', default='./data/train.BPE.ha', type=str, required=False, help='')
    parser.add_argument('--dev_source_dataset', default='./data/dev.BPE.en', type=str, required=False, help='')
    parser.add_argument('--dev_target_dataset', default='./data/dev.BPE.ha', type=str, required=False, help='')
    
    parser.add_argument('--train_file', default='./data/train.pkl', type=str, required=False, help='')
    parser.add_argument('--dev_file', default='./data/dev.pkl', type=str, required=False, help='')

    parser.add_argument('--source_vocab', default='./data/joint_vocab.json', type=str, required=False, help='')
    parser.add_argument('--target_vocab', default='./data/joint_vocab.json', type=str, required=False, help='')

    parser.add_argument('--max_src_len', default=128, type=int, required=False, help='')
    parser.add_argument('--max_trg_len', default=128, type=int, required=False, help='')

    parser.add_argument('--enc_emb_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--dec_emb_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--enc_hid_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--dec_hid_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--dec_out_dim', default=64, type=int, required=False, help='')
    parser.add_argument('--enc_n_head', default=8, type=int, required=False, help='')
    parser.add_argument('--enc_n_layer', default=2, type=int, required=False, help='')
    parser.add_argument('--dec_n_layer', default=2, type=int, required=False, help='')
    parser.add_argument('--enc_dropout', default=0.1, type=float, required=False, help='')
    parser.add_argument('--dec_dropout', default=0.3, type=float, required=False, help='')
    
    parser.add_argument('--num_workers', default=8, type=int, required=False, help='')
    parser.add_argument('--shuffle', default=True, type=bool, required=False, help='whether to shuffle the training dataset when loading')

    parser.add_argument('--evaluate_step', default=100, type=int, required=False, help='')

    parser.add_argument('--epochs', default=50, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='')
    parser.add_argument('--lr', default=1.0e-03, type=float, required=False, help='learning rate')
    parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='')
    parser.add_argument('--weight_decay', default=1.0e-04, type=float, required=False, help='')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--patience', type=int, default=0, help='for early stopping')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='')
    parser.add_argument('--seed', type=int, default=1234, help='')
    args = parser.parse_args()
    return args


def create_logger(args):
    """
    logger.info the log in console and log files
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
                
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = src
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)

     

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
                
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
                
        #input = [1, batch size]
        
        embedded = input.unsqueeze(0)
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, emb_dim, dropout, encoder, decoder, device):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        src = self.dropout(self.embedding(src)) #src = [src_len, batch_size, emb_dim]
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            input = self.dropout(self.embedding(input))
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn(batch):
    src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
    src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.bool)
    trg_ids = torch.tensor([example['trg_ids'] for example in batch], dtype=torch.long)
    trg_mask = torch.tensor([example['trg_mask'] for example in batch], dtype=torch.bool)
    
    return {"src_ids": src_ids,
            "src_mask": src_mask,
            "trg_ids": trg_ids,
            "trg_mask": trg_mask
            }


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        src_ids, src_mask, trg_ids, trg_mask = self.data[index]
        return {"src_ids": src_ids,
                "src_mask": src_mask,
                "trg_ids": trg_ids,
                "trg_mask": trg_mask,
                }

    def __len__(self):
        return len(self.data)



def load_dataset(logger, args):

    logger.info("loading training dataset and validating dataset")

    with open(args.train_file, "rb") as f:
        train_data = pickle.load(f)
    with open(args.dev_file, "rb") as f:
        dev_data = pickle.load(f)

    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(dev_data)

    return train_dataset, val_dataset


def train(args, logger, model, train_iterator, valid_iterator, optimizer, criterion, iter_loss_list):
    
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(train_iterator):
    
        src = batch['src_ids'].to(args.device)
        src_mask = batch['src_mask'].to(args.device)
        trg = batch['trg_ids'].to(args.device)
        trg_mask = batch['trg_mask'].to(args.device)

        src = torch.transpose(src, 1, 0) # src = [src_len, batch_size]
        trg = torch.transpose(trg, 1, 0) # trg = [trg_len, batch_size]

        optimizer.zero_grad()
        
        output = model(src, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        
        # trg = trg[1:].view(-1)
        trg = trg[1:].contiguous().view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        
        loss = loss.item()
        if (i+1) % args.evaluate_step == 0:
            valid_loss = evaluate(args, logger, model, valid_iterator, criterion)
             # when evaluating, the mode is changed to model.eval(), so we need to change it back to model.train()
            model.train()

            logger.info(f"Train iter:{i+1}, Train loss:{loss: .3f}, Val. loss:{valid_loss: .3f}")

            # # record the training and valid loss each evaluate iteration, and then plot the curve
            # iter_loss_list.append((loss, valid_loss))
            # with open(args.iter_save_file_path, "w") as f:
            #     json.dump(iter_loss_list, f)
            # plot_curve(args, "iter")

        epoch_loss += loss
        
    return epoch_loss / len(train_iterator), iter_loss_list


def evaluate(args, logger, model, iterator, criterion):
    
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            
            src = batch['src_ids'].to(args.device) # src = [batch_size, src_len]
            src_mask = batch['src_mask'].to(args.device) # src_mask (src_padding_mask) = [batch_size, src_len]
            trg = batch['trg_ids'].to(args.device) # trg_mask = [batch_size, trg_len]

            src = torch.transpose(src, 1, 0) # src = [src_len, batch_size]
            trg = torch.transpose(trg, 1, 0) # trg = [trg_len, batch_size]

            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)

            # trg = trg[1:].view(-1)
            trg = trg[1:].contiguous().view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():

    args = set_args()
    logger = create_logger(args)

    create_data(args, logger)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    # save the training and valid loss to plot training curve
    args.iter_save_file_path = f"./loss/Fairseq_shuffle{str(args.shuffle)}_iter{args.evaluate_step}_bs{args.batch_size}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.json"
    args.iter_save_pic_path = f"./pic/Fairseq_shuffle{str(args.shuffle)}_iter{args.evaluate_step}_bs{args.batch_size}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.jpg"
    args.epoch_save_file_path = f"./loss/Fairseq_shuffle{str(args.shuffle)}_epoch_bs{args.batch_size}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.json"
    args.epoch_save_pic_path = f"./pic/Fairseq_shuffle{str(args.shuffle)}_epoch_bs{args.batch_size}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.jpg"
    args.model_save_path = f"./model/Fairseq_shuffle{str(args.shuffle)}_epoch_bs{args.batch_size}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.pt"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    train_dataset, validate_dataset = load_dataset(logger, args)
    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    valid_iterator = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    
    with open(args.source_vocab) as f:
        source_vocab = json.load(f)
    with open(args.target_vocab) as f:
        target_vocab = json.load(f)
    
    PAD = "<PAD>"
    SRC_PAD_IDX = source_vocab[PAD]
    TRG_PAD_IDX = target_vocab[PAD]

    INPUT_DIM = len(source_vocab)
    OUTPUT_DIM = len(target_vocab)

    ENC_EMB_DIM = args.enc_emb_dim
    DEC_EMB_DIM = args.dec_emb_dim
    ENC_HID_DIM = args.enc_hid_dim
    DEC_HID_DIM = args.dec_hid_dim
    ENC_DROPOUT = args.enc_dropout
    DEC_DROPOUT = args.dec_dropout

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(INPUT_DIM, ENC_EMB_DIM, DEC_DROPOUT, enc, dec, device).to(device)
    model.apply(init_weights)
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    best_valid_loss = float('inf')

    iter_loss_list = []
    epoch_loss_list = []

    for epoch in range(args.epochs):
        
        start_time = time.time()

        train_loss, iter_loss_list = train(args, logger, model, train_iterator, valid_iterator, optimizer, criterion,  iter_loss_list)
        valid_loss = evaluate(args, logger, model, valid_iterator, criterion)

        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # save losses in json and plot curve 
        # epoch_loss_list.append((train_loss, valid_loss))
        # with open(args.epoch_save_file_path, "w") as f:
        #     json.dump(epoch_loss_list, f)
        # plot_curve(args, "epoch")

        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), args.model_save_path)
        
        logger.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logger.info(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    main()
