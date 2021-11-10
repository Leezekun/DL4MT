import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
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
    parser.add_argument('--device', default='2', type=str, required=False, help='')
    parser.add_argument('--no_cuda', action='store_true', help='')
    parser.add_argument('--log_path', default='./log/train.log', type=str, required=False, help='')

    parser.add_argument('--train_source_dataset', default='./data/train.BPE.en', type=str, required=False, help='')
    parser.add_argument('--train_target_dataset', default='./data/train.BPE.ha', type=str, required=False, help='')
    parser.add_argument('--dev_source_dataset', default='./data/dev.BPE.en', type=str, required=False, help='')
    parser.add_argument('--dev_target_dataset', default='./data/dev.BPE.ha', type=str, required=False, help='')
    
    parser.add_argument('--pretrained_model', default='./model/best_whole_model/hybridnmt.pt', type=str, required=False, help='')

    parser.add_argument('--train_file', default='./data/train.pkl', type=str, required=False, help='')
    parser.add_argument('--dev_file', default='./data/dev.pkl', type=str, required=False, help='')

    parser.add_argument('--source_vocab', default='./data/joint_vocab.json', type=str, required=False, help='')
    parser.add_argument('--target_vocab', default='./data/joint_vocab.json', type=str, required=False, help='')

    parser.add_argument('--max_src_len', default=128, type=int, required=False, help='')
    parser.add_argument('--max_trg_len', default=128, type=int, required=False, help='')

    parser.add_argument('--enc_emb_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--dec_emb_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--enc_hid_dim', default=1024, type=int, required=False, help='')
    parser.add_argument('--dec_hid_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--dec_out_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--enc_n_head', default=8, type=int, required=False, help='')
    parser.add_argument('--enc_n_layer', default=2, type=int, required=False, help='')
    parser.add_argument('--dec_n_layer', default=2, type=int, required=False, help='')
    parser.add_argument('--enc_dropout', default=0.1, type=float, required=False, help='')
    parser.add_argument('--dec_dropout', default=0.1, type=float, required=False, help='')

    parser.add_argument('--num_workers', default=8, type=int, required=False, help='')
    parser.add_argument('--shuffle', default=True, type=bool, required=False, help='whether to shuffle the training dataset when loading')

    parser.add_argument('--evaluate_step', default=200, type=int, required=False, help='')

    parser.add_argument('--epochs', default=50, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=512, type=int, required=False, help='')
    parser.add_argument('--lr', default=1.0e-03, type=float, required=False, help='learning rate')
    parser.add_argument('--eps', default=1.0e-06, type=float, required=False, help='')
    parser.add_argument('--weight_decay', default=1.0e-06, type=float, required=False, help='')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--patience', type=int, default=0, help='for early stopping')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='')
    parser.add_argument('--seed', type=int, default=1234, help='')

    parser.add_argument('--pretrain', default=True, type=bool, required=False, help='')

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


class TFEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, dim_feedforward, n_head, n_layer, dropout):
        super().__init__()
        
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layers = TransformerEncoderLayer(emb_dim, n_head, dim_feedforward, dropout, activation="gelu")
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

    def forward(self, src, src_padding_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, emb_dim]
            src_mask: Tensor, shape [seq_len, seq_len] (optional)
            src_padding_mask: Tensor, shape [batch_size, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, input_dim]
        """
        
        src = self.pos_encoder(src) #src = [src_len, batch_size, emb_dim]
        src = self.layer_norm(src)
        src = self.dropout(src)

        # outputs = self.transformer_encoder(src, src_mask) #outputs = [src_len, batch_size, emb_dim]

        outputs = self.transformer_encoder(src=src, src_key_padding_mask=src_padding_mask)
        
        # outputs = self.transformer_encoder(src=src)

        return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embed_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class Attention(nn.Module):
    def __init__(self, enc_out_dim, emb_dim, dec_hid_dim, dec_out_dim):
        super().__init__()
        
        # self.attn = nn.Linear(enc_out_dim + dec_hid_dim, dec_hid_dim)
        # self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        self.input_proj = nn.Linear(dec_hid_dim, enc_out_dim)
        self.output_proj = nn.Linear(dec_hid_dim + enc_out_dim, dec_out_dim)

    def forward(self, hidden, encoder_outputs, encoder_padding_mask):
        # encoder_outputs = [src_len, batch_size, enc_output_dim]
        # hidden = [batch_size, dec_hid_dim]

        src_len = encoder_outputs.shape[0]
        
        # hidden_ = hidden.unsqueeze(1).repeat(1, src_len, 1) # hidden = [batch_size, src_len, dec_hid_dim]
        # encoder_outputs = encoder_outputs.permute(1, 0, 2) # encoder_outputs = [batch_size, src_len, enc_output_dim]
        # energy = torch.tanh(self.attn(torch.cat((hidden_, encoder_outputs), dim = 2))) # [batch_size, src_len, dec_hid_dim]
        # energy = self.dropout(energy)
        # attention = self.v(energy).squeeze() # [batch_size, src_len]
        hidden = self.input_proj(hidden)
        proj_hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # hidden = [batch_size, src_len, enc_out_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # encoder_outputs = [batch_size, src_len, enc_output_dim]
        attention = (encoder_outputs * proj_hidden).sum(dim=2) # [batch_size, src_len]
        attention = (
                attention.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attention)
            )
        attention = F.softmax(attention, dim=1) # [batch_size, src_len]
        weighted = (encoder_outputs * attention.unsqueeze(2)).sum(dim=1) # [batch_size, enc_output_dim]
        output = torch.tanh(self.output_proj(torch.cat((weighted, hidden), dim=1))) # # [batch_size, dec_out_dim]
        return output


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_out_dim, dec_hid_dim, dec_out_dim, dec_n_layer, dropout, attention):
        super().__init__()

        self.model_type = "LSTM"

        self.output_dim = output_dim
        self.attention = attention
        self.dec_hid_dim = dec_hid_dim
        self.dec_out_dim = dec_out_dim
        self.emb_dim = emb_dim
        
        self.rnn1 = nn.LSTMCell(dec_out_dim + emb_dim, dec_hid_dim)
        self.rnn2 = nn.LSTMCell(dec_hid_dim, dec_hid_dim)

        self.fc_out = nn.Linear(dec_out_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

    def forward(self, input, input_feed, hidden1, cell1, hidden2, cell2, encoder_outputs, encoder_padding_mask):
        
        # hidden1, hidden2 = [batch size, dec hid dim]
        
        embedded = self.layer_norm(input)
        embedded = self.dropout(embedded) #embedded = [batch size, emb_dim]

        rnn1_input = torch.cat((embedded, input_feed), dim = 1) #embedded = [batch size, emb_dim + dec_hid_dim]
    
        hidden1, cell1 = self.rnn1(rnn1_input, (hidden1, cell1))
    
        rnn2_input = hidden1
        hidden2, cell2 = self.rnn2(rnn2_input, (hidden2, cell2))
        
        input_feed = self.attention(hidden2, encoder_outputs, encoder_padding_mask) # [batch_size, dec_out_dim]
        input_feed = self.dropout(input_feed)
        prediction = self.fc_out(torch.cat((self.dropout(input_feed), embedded), 1))
        return prediction, hidden1, cell1, hidden2, cell2, input_feed


class HybridNMT(nn.Module):
    def __init__(self, input_dim, emb_dim, dropout, encoder, decoder, device):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask, trg, teacher_forcing_ratio = 1):
       
        batch_size = src.shape[1] 
        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        dec_hid_dim = self.decoder.dec_hid_dim
        dec_out_dim = self.decoder.dec_out_dim
        
        # initial hidden vector for n_layer lstm
        hidden1 = hidden2 = cell1 = cell2 = torch.zeros(batch_size, dec_hid_dim).to(self.device)
        input_feed = torch.zeros(batch_size, dec_out_dim).to(self.device)

        #tensor to store decoder outputs
        outputs = torch.ones(trg_len, batch_size, trg_vocab_size).to(self.device) 
        predicts = torch.ones(trg_len, batch_size).to(self.device) 

        #src = [src_len, batch_size]
        src = self.embedding(src) * math.sqrt(self.emb_dim) #src = [src_len, batch_size, emb_dim]
        # src = self.dropout(src)
        encoder_outputs = self.encoder(src, src_mask) #encoder_outputs = [src_len, batch_size, emb_dim] 

        #first input to the decoder is the <go> tokens
        input = trg[0,:] #input = [trg_len, batch_size]
        for t in range(1, trg_len):
            input = self.embedding(input)
            output, hidden1, cell1, hidden2, cell2, input_feed = self.decoder(input, input_feed, hidden1, cell1, hidden2, cell2, encoder_outputs, src_mask)
            outputs[t] = output
        
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            predicts[t] = top1
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs, predicts


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
    epoch_acc = 0
    for i, batch in enumerate(train_iterator):
    
        src = batch['src_ids'].to(args.device)
        src_mask = batch['src_mask'].to(args.device)
        trg = batch['trg_ids'].to(args.device)
        trg_mask = batch['trg_mask'].to(args.device)

        src = torch.transpose(src, 1, 0) # src = [src_len, batch_size]
        trg = torch.transpose(trg, 1, 0) # trg = [trg_len, batch_size]

        optimizer.zero_grad()
        
        output, predict = model(src, src_mask, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]

        correct = predict.eq(trg).sum(0) / (~trg_mask).sum(1)
        acc = correct.mean()

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
        acc = acc.item()
        if (i+1) % args.evaluate_step == 0:
            valid_loss, valid_acc = evaluate(args, logger, model, valid_iterator, criterion)
             # when evaluating, the mode is changed to model.eval(), so we need to change it back to model.train()
            model.train()

            logger.info(f"Train iter:{i+1}, Train loss:{loss: .3f}, acc:{acc: .3f}, Val. loss:{valid_loss: .3f}, acc:{valid_acc: .3f},")

            # record the training and valid loss each evaluate iteration, and then plot the curve
            iter_loss_list.append((loss, valid_loss))
            with open(args.iter_save_file_path, "w") as f:
                json.dump(iter_loss_list, f)
            plot_curve(args, "iter")

        epoch_loss += loss
        epoch_acc += acc
        
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator), iter_loss_list


def evaluate(args, logger, model, iterator, criterion):
    
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            
            src = batch['src_ids'].to(args.device) # src = [batch_size, src_len]
            src_mask = batch['src_mask'].to(args.device) # src_mask (src_padding_mask) = [batch_size, src_len]
            trg = batch['trg_ids'].to(args.device) # trg_mask = [batch_size, trg_len]
            trg_mask = batch['trg_mask'].to(args.device)

            src = torch.transpose(src, 1, 0) # src = [src_len, batch_size]
            trg = torch.transpose(trg, 1, 0) # trg = [trg_len, batch_size]

            output, predict = model(src, src_mask, trg) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
 
            correct = predict.eq(trg).sum(0) / (~trg_mask).sum(1)
            acc = correct.mean()

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)

            # trg = trg[1:].view(-1)
            trg = trg[1:].contiguous().view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


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
    args.iter_save_file_path = f"./loss/iter{args.evaluate_step}_bs{args.batch_size}_decout{args.dec_out_dim}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.json"
    args.iter_save_pic_path = f"./pic/iter{args.evaluate_step}_bs{args.batch_size}_decout{args.dec_out_dim}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.jpg"
    args.epoch_save_file_path = f"./loss/epoch_bs{args.batch_size}_decout{args.dec_out_dim}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.json"
    args.epoch_save_pic_path = f"./pic/epoch_bs{args.batch_size}_decout{args.dec_out_dim}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.jpg"
    args.model_save_path = f"./model/whole_model_epoch_bs{args.batch_size}_decout{args.dec_out_dim}_loss_lr{args.lr}_wd{args.weight_decay}_eps{args.eps}_enc_dp{args.enc_dropout}_dec_dp{args.dec_dropout}.pt"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    train_dataset, validate_dataset = load_dataset(logger, args)
    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    valid_iterator = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    
    # make sure using a joint vocab for the source and target language 
    assert args.source_vocab == args.target_vocab
    with open(args.source_vocab) as f:
        vocab = json.load(f)
    
    PAD = "<PAD>"
    PAD_IDX = vocab[PAD]

    INPUT_DIM = OUTPUT_DIM = len(vocab)
    # make sure using a joint vocab and embedding for the source and target language 
    assert args.enc_emb_dim == args.dec_emb_dim
    EMB_DIM = ENC_EMB_DIM = DEC_EMB_DIM = args.enc_emb_dim
    ENC_HID_DIM = args.enc_hid_dim
    DEC_HID_DIM = args.dec_hid_dim
    DEC_OUT_DIM = args.dec_out_dim
    ENC_N_HEAD = args.enc_n_head
    ENC_N_LAYER = args.enc_n_layer
    DEC_N_LAYER = args.dec_n_layer
    ENC_DROPOUT = args.enc_dropout
    DEC_DROPOUT = args.dec_dropout

    attn = Attention(ENC_EMB_DIM, DEC_EMB_DIM, DEC_HID_DIM, DEC_OUT_DIM)
    encoder = TFEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, ENC_N_HEAD, ENC_N_LAYER, ENC_DROPOUT)
    decoder = LSTMDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_EMB_DIM, DEC_HID_DIM, DEC_OUT_DIM, DEC_N_LAYER, DEC_DROPOUT, attn)

    model = HybridNMT(INPUT_DIM, EMB_DIM, DEC_DROPOUT, encoder, decoder, device).to(device)
    if args.pretrain: 
        # model = torch.load(args.pretrained_model)
        model = torch.load(args.model_save_path)
        # model.load_state_dict(torch.load(args.model_save_path))

    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    best_valid_loss = float('inf')

    iter_loss_list = []
    epoch_loss_list = []

    for epoch in range(args.epochs):
        
        start_time = time.time()

        train_loss, train_acc, iter_loss_list = train(args, logger, model, train_iterator, valid_iterator, optimizer, criterion, iter_loss_list)
        valid_loss, valid_acc = evaluate(args, logger, model, valid_iterator, criterion)

        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # save losses in json and plot curve 
        epoch_loss_list.append((train_loss, valid_loss))
        with open(args.epoch_save_file_path, "w") as f:
            json.dump(epoch_loss_list, f)
        plot_curve(args, "epoch")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), args.model_save_path)
            torch.save(model, args.model_save_path)

        logger.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logger.info(f'\tVal. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc:.3f}|  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    main()
