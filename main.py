from utils import (
    basic_tokenizer, word2char, count_parameters, translate_sentence,
    save_model, load_model
)
from models import Encoder, Decoder, Seq2Seq
from pipeline import train, evaluate
from metrics import evaluation_report

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import os
import warnings as wrn
wrn.filterwarnings('ignore')


def main():
    df = pd.read_csv('./Dataset/sec_dataset_II.csv')
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:, [1, 0]]
    df = df.iloc[:, :]

    train_df, test_df = train_test_split(df, test_size=.15)
    train_df, valid_df = train_test_split(train_df, test_size=.05)
    # print(len(train_df), len(valid_df), len(test_df))
    train_df.to_csv('./Dataset/train.csv', index=False)
    valid_df.to_csv('./Dataset/valid.csv', index=False)
    test_df.to_csv('./Dataset/test.csv', index=False)

    SRC = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    TRG = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    fields = {
        'Error': ('src', SRC),
        'Word': ('trg', TRG)
    }
    train_data, valid_data, test_data = TabularDataset.splits(
        path='./Dataset',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields
    )
    SRC.build_vocab(train_data, min_freq=100)
    TRG.build_vocab(train_data, min_freq=50)
    # print(len(SRC.vocab), len(TRG.vocab))

    # Hyperparameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 256
    #
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    EMB_DIM = 64  # 64
    HID_DIM = 128  # 256 # each conv. layer has 2 * hid_dim filters
    ENC_LAYERS = 10  # number of conv. blocks in encoder
    DEC_LAYERS = 10  # number of conv. blocks in decoder
    ENC_KERNEL_SIZE = 3  # must be odd!
    DEC_KERNEL_SIZE = 3  # can be even or odd
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    CLIP = 0.1
    PATH = './Checkpoints/conv_s2s.pth'

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )

    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, DEVICE)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, DEVICE)
    model = Seq2Seq(enc, dec).to(DEVICE)
    # print(model)
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    epoch = 1
    # load the model
    if os.path.exists(PATH):
        checkpoint, epoch, train_loss = load_model(model, PATH)
    #
    N_EPOCHS = epoch + 100
    best_loss = 1e10

    for epoch in range(epoch, N_EPOCHS):
        print(f"Epoch: {epoch} / {N_EPOCHS}")
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        print(f"Train Loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, train_loss, epoch, PATH)

    # example_idx = 10
    # src = vars(train_data.examples[example_idx])['src']
    # trg = vars(train_data.examples[example_idx])['trg']
    # print(f'src = {src}')
    # print(f'trg = {trg}')
    # translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)
    # print(f'predicted trg = {translation}')

    evaluation_report(valid_data, SRC, TRG, model, DEVICE)


if __name__ == '__main__':
    main()
