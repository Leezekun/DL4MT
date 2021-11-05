import re
import matplotlib
import matplotlib.pyplot as plt  
import numpy as np  
import json

def plot_loss(save_path, 
                x_vals, y_vals, 
                x_label, y_label, 
                x2_vals=None, y2_vals=None, 
                legend=None,
                figsize=(3.5, 2.5)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals is not None and y2_vals is not None:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
    if legend:
        plt.legend(legend)
    plt.savefig(save_path)
    plt.close()

def load_loss(file_path):
    with open(file_path, 'r') as f:
        losses = json.load(f)

    train_losses = []
    valid_losses = []
    for train_loss, valid_loss in losses:
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    return train_losses, valid_losses


def plot_curve(args, mode):
    if mode == "iter":
        iter_train_losses, iter_valid_losses = load_loss(args.iter_save_file_path)
        iter_x = np.arange(len(iter_train_losses))*args.evaluate_step

        plot_loss(args.iter_save_pic_path,
                    iter_x, iter_train_losses,
                    "iteration", "loss",
                    iter_x, iter_valid_losses,
                    ["train", "valid"]
                    )

    elif mode == "epoch":
        epoch_train_losses, epoch_valid_losses = load_loss(args.epoch_save_file_path)
        epoch_x = np.arange(len(epoch_train_losses))

        plot_loss(args.epoch_save_pic_path,
                    epoch_x, epoch_train_losses,
                    "epoch", "loss",
                    epoch_x, epoch_valid_losses,
                    ["train", "valid"]
                    )
    else:
        print("Wrong mode!")


def plot_noise_distribution():
    with open("./data/batch_noise_data.json", 'r') as f:
        noise = json.load(f)

    batch_return = []
    batch_brackets = []
    for batch_contain_return, _, batch_contain_brackets in noise:
        batch_return.append(batch_contain_return)
        batch_brackets.append(batch_contain_brackets)
    x = np.arange(len(noise))
    plot_loss("./pic/noise_distribution.jpg",
                    x, batch_brackets,
                    "batch", "#noise"
                    )


def plot_no_shuffle():
    with open("./data/no_shuffle.txt", 'r') as f:
        noises = f.readlines()

    batch_train = []
    batch_valid = []
    for noise in noises:
        noise = noise.split(",")
        train_loss = float(noise[0].strip())
        valid_loss = float(noise[1].strip())
        batch_train.append(train_loss)
        batch_valid.append(valid_loss)
    x = np.arange(len(noises))*100
    plot_loss("./pic/no_shuffle.jpg",
                    x, batch_train,
                    "batch", "loss",
                    x, batch_valid,
                    ["train", "valid"],
                    )

if __name__ == '__main__':
    # plot_noise_distribution()
    plot_no_shuffle()
    
        
