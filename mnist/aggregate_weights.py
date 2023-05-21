import scipy
import torch
import numpy as np


def trimmed_mean_weights(params, beta):
    return scipy.stats.trim_mean(params, proportiontocut=beta, axis=0)

    # scores = np.asarray(scores)
    # grads = torch.stack(grads)
    # mask = (scores == 1)
    #
    # if sum(mask) == 0:
    #     print('sum(mask) = 0')
    #     tmp = torch.zeros_like(grads[0])
    # elif sum(mask) == 1:
    #     print('sum(mask) = 1, ', grads[mask])
    #     tmp = grads[mask][0]
    # else:  # sum(mask) > 1:
    #     # remove beta percent data and then compute the trimmed mean
    #     d = grads.shape[1] # dimensions
    #     tmp = torch.zeros_like(grads[0])
    #     for i in range(d): # for each dimension
    #         ts = sorted([(v, v[i]) for v in grads[mask]], key=lambda vs: vs[1], reverse=False)
    #         m = int(np.floor(len(ts) * beta))  # remove the m smallest and m biggest grads from ts
    #         if 2 * m >= len(ts):
    #             raise ValueError(f'beta({beta}) is too large!')
    #         elif m == 0:
    #             # print(i, d, grads[mask][:, i], flush=True)
    #             _tmp = torch.mean(grads[mask][:, i], dim=0)
    #         else:
    #             _tmp = torch.mean(torch.stack([vs[0][i] for vs in ts[m:-m]]), dim=0)
    #         tmp[i] = _tmp

    # # remove beta percent data and then compute the trimmed mean
    # d = params.shape[1] # dimensions
    # tmp = torch.zeros_like(params[0])
    # for i in range(d): # for each dimension
    #     ts = sorted([(v, v[i]) for v in params], key=lambda vs: vs[1], reverse=False)
    #     m = int(np.floor(len(ts) * beta))  # remove the m smallest and m biggest grads from ts
    #     if 2 * m >= len(ts):
    #         raise ValueError(f'beta({beta}) is too large!')
    #     elif m == 0:
    #         # print(i, d, grads[mask][:, i], flush=True)
    #         _tmp = torch.mean(params[:, i], dim=0)
    #     else:
    #         _tmp = torch.mean(torch.stack([vs[0][i] for vs in ts[m:-m]]), dim=0)
    #     tmp[i] = _tmp
    #
    # return tmp




def plot_data(data_lst, cluster_assignment, is_show=False):
    import matplotlib.pyplot as plt
    # plot X
    colors = ['r', 'g', 'b', 'black', 'm', 'brown', 'purple', 'yellow',
              'tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray',
              'tab:olive','tab:cyan',]
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    j = 0
    cnt = 0
    title = ''
    for i, (X, y, y_label) in enumerate(data_lst):
        if i == 0:
            label = y_label[0]
        elif i > 0 and cluster_assignment[i - 1] != cluster_assignment[i]:
            title += f'{cnt}, '
            j += 1
            # print(j)
            label = y_label[0]
            cnt = 0
        else:
            label = None
        if label is not None: title += f'{label}:'
        cnt += len(X)
        # print(f'j:{j}', colors[j], flush=True)
        plt.scatter(X[:, 0], X[:, 1], c=[colors[j%len(colors)]] * len(y), label=label, alpha=0.5)
    title += f'{cnt}'
    plt.xlabel(f'X[:, 0]')
    plt.ylabel(f'X[:, 1]')
    plt.title(title)
    plt.legend()
    if is_show: plt.show()
    plt.clf()

    # plot X[:, 0] and y
    for idx in range(X.shape[1]):
        j = 0
        for i, (X, y, y_label) in enumerate(data_lst):
            if i == 0:
                label = y_label[0]
            elif i > 0 and cluster_assignment[i - 1] != cluster_assignment[i]:
                j += 1
                # print(j)
                label = y_label[0]
            else:
                label = None
            plt.scatter(X[:, idx], y, c=[colors[j%len(colors)]] * len(y), label=label, alpha=0.5)
        plt.xlabel(f'X[:, {idx}]')
        plt.ylabel(f'y')
        plt.legend()
        if is_show: plt.show()
        plt.clf()

