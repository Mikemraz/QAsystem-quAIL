"""Train a model on SQuAD.
Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from torch.utils.data import DataLoader
import utils.util as util
from sklearn.metrics import accuracy_score


from collections import OrderedDict
from json import dumps
from baseline_model import BiDAF
from utils.util import get_dataset

from tqdm import tqdm
from ujson import load as json_load
import warnings
warnings.filterwarnings("ignore")


def main():
    # Set up logging and devices
    save_dir = "./save/my_training.pt"
    load_path = "./save/my_training.pt"
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    # Set random seed
    ramdom_seed = 123
    random.seed(ramdom_seed)
    np.random.seed(ramdom_seed)
    torch.manual_seed(ramdom_seed)
    torch.cuda.manual_seed_all(ramdom_seed)

    # get processed batch dataloader
    # build standard Pytorch DataLoader object of our data
    train_dataset, dev_dataset = get_dataset()
    dataloader_train = DataLoader(train_dataset, shuffle=True, batch_size=64)
    dataloader_dev = DataLoader(dev_dataset, shuffle=True, batch_size=64)

    # define model parameters
    drop_prob = 0.0
    embedding_size = 32
    vocab_size = len(train_dataset.vocab)
    hidden_size = 128
    lr = 0.5
    l2_wd = 0
    max_grad_norm = 5.0

    # Get model
    model = BiDAF(embedding_size=embedding_size,
                  vocab_size=vocab_size,
                  hidden_size=hidden_size,
                  drop_prob=drop_prob)

    if load_path:
        model, step = util.load_model(model, load_path, device)
    else:
        step = 0
    model = model.to(device)
    model.train()

    # training parameters
    ema = util.EMA(model, 0.999)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    num_epochs = 4

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), lr,
                               weight_decay=l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    # log.info('Building dataset...')
    # train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    # train_loader = data.DataLoader(train_dataset,
    #                                batch_size=args.batch_size,
    #                                shuffle=True,
    #                                num_workers=args.num_workers,
    #                                collate_fn=collate_fn)
    # dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    # dev_loader = data.DataLoader(dev_dataset,
    #                              batch_size=args.batch_size,
    #                              shuffle=False,
    #                              num_workers=args.num_workers,
    #                              collate_fn=collate_fn)


    epoch = step // len(train_dataset)
    while epoch != num_epochs:
        epoch += 1
        epoch_losses = []
        print("epoch {}:".format(epoch))
        for cw_idxs, qw_idxs, y in dataloader_train:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)
            optimizer.zero_grad()

            # Forward
            logits = model(cw_idxs, qw_idxs)
            y = y.to(device)
            loss = criterion(logits,y)
            loss_val = loss.item()
            # print("batch loss: {}".format(loss_val))
            epoch_losses.append(loss_val)
            # Backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step(step // batch_size)
            ema(model, step // batch_size)

            step += batch_size

        epoch_loss = np.mean(epoch_losses)
        print("training loss: {}".format(epoch_loss))
        ema.assign(model)
        y_true_train, y_pred_train = predict(model, dataloader_train, device)
        y_true_dev, y_pred_dev = predict(model, dataloader_dev, device)
        ema.resume(model)
        acc_train = accuracy_score(y_true_train, y_pred_train)
        acc_test = accuracy_score(y_true_dev, y_pred_dev)
        print("train accuracy: {0}, dev accuracy: {1}\n".format(acc_train,acc_test))

    torch.save(model.state_dict(), save_dir)


def predict(model, data_loader, device):
    y_pred = []
    y_true = []

    for i, batch in enumerate(data_loader):
        cw_idxs, qw_idxs, y = batch
        cw_idxs = cw_idxs.to(device)
        qw_idxs = qw_idxs.to(device)
        # Forward
        logits = model(cw_idxs, qw_idxs)

        y_pred_batch = torch.argmax(logits, 1)
        y_pred_batch = y_pred_batch.cpu()
        y_true_batch = y
        y_pred.append(y_pred_batch)
        y_true.append(y_true_batch)

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    return y_true, y_pred




if __name__ == '__main__':
    main()