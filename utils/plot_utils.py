import matplotlib.pyplot as plt
import numpy as np


def plot_raw(clean, adv, file_name, is_norm=False):
    if is_norm:
        max_, min_ = np.max(clean), np.min(clean)
        clean = (clean - min_) / (max_ - min_)
        adv = (adv - min_) / (max_ - min_)

    plt.figure()
    x = np.arange(clean.shape[1]) * 1.0 / 256
    l1, = plt.plot(x,
                   adv[0] - np.mean(adv[0]),
                   linewidth=2.0,
                   color='red',
                   label='Adversarial sample')  # plot adv data
    l2, = plt.plot(x,
                   clean[0] - np.mean(adv[0]),
                   linewidth=2.0,
                   color='dodgerblue',
                   label='Original sample')  # plot clean data
    for i in range(1, 5):
        plt.plot(x, adv[i] + i - np.mean(adv[i]), linewidth=2.0,
                 color='red')  # plot adv data
        plt.plot(x,
                 clean[i] + i - np.mean(adv[i]),
                 linewidth=2.0,
                 color='dodgerblue')  # plot clean data

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-0.5, 5.0])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(handles=[l2, l1],
               labels=['Original sample', 'Poisoned sample'],
               loc='upper right',
               ncol=2,
               fontsize=10)
    plt.savefig(file_name + '.png', dpi=300)


def plot_mask_dist(mask_params):
    mask_list = []
    for param in mask_params:
        mask_list.extend(param.cpu().data.flatten().tolist())
    fig = plt.figure(figsize=(4, 3))
    plt.hist(mask_list, bins=50, ec='k')
    plt.savefig('fig/pruning_mask.png', dpi=300)