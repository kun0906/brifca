import os.path
import pickle

import copy
import numpy as np

from synthetic.process_runner import product_dict


def get_CFG(p=[5], m=[200], alg_method=[], update_method=[]):
    # Note that the last three parameters must be "data_seed, train_seed, and lr"
    CFG = {
        "p": p,  # number of distributions/clusters

        "m": m,  # number of total machines (Normal + Byzantine)
        'alpha': [0.05],  # percent of Byzantine machines

        "n": [100],  # [50, 100],  # number of data points per each machine, [50, 100, 200, 400, 800]

        "d": [20, 50, 100, 200, 500],  # different data dimensions: [5, 25, 50, 100, 200]

        "noise_scale": [0.4472],  # standard deviation of noise/epsilon: sigma**2 = 0.2

        "r": [1.0],  # separation parameter for synthetic data generation

        "alg_method": alg_method,  # ['baseline', 'proposed'],

        'update_method': update_method,  # gradient update methods for server
        'beta': [0.05],  # trimmed means parameters

        "data_seed": [v for v in range(0, 100, 2)],  # different seeds for data

        "train_seed": [0],  # different seeds for training

        'lr': [0.01],  # different learning rates
    }

    return CFG


def extract_data(results, cfg2, plot_metric='min_dist'):
    # plot_metric = 'min_dist'    # 'max_dist'    # min_loss, 'min_dist'
    ms = cfg2['m']
    ds = cfg2['d']
    ns = cfg2['n']
    ps = cfg2['p']
    plot_data = {}
    # print(f"m {cfg2['m'][0]}, r {cfg2['r'][0]}")
    print(f"n {cfg2['n'][0]}")

    alg_methods = cfg2['alg_method']
    update_methods = cfg2['update_method']
    xs = ds  # x-axis
    # file_name = 'ds'
    # x_label = 'Dim'
    # print(f'plot_metric: {plot_metric}, xs: {xs}, update_methods: {update_methods}')

    data_seeds = cfg2['data_seed']
    train_seeds = cfg2['train_seed']
    lrs = cfg2['lr']
    del cfg2['data_seed']
    del cfg2['train_seed']
    del cfg2['lr']
    # del cfg2['alg_method']

    cfgs2 = list(product_dict(**cfg2))

    plot_data = {}
    for alg_method in alg_methods:
        for update_method in update_methods:
            ys = []
            ys_erros = []
            for m in xs:  # x_labels: different machines, y_label: metric
                for cfg in cfgs2:  # all results
                    key1 = cfg.keys()
                    key1v = cfg.values()
                    # if cfg['m'] != m or cfg['update_method'] != update_method: continue
                    if cfg['d'] != m or cfg['update_method'] != update_method or cfg[
                        'alg_method'] != alg_method: continue
                    # THRE = THRE0 * cfg["noise_scale"]
                    success_rate = 0
                    vs = []
                    for d_i in data_seeds:
                        for t_i in train_seeds:
                            for lr in lrs:
                                key2 = tuple(list(key1v) + [d_i, t_i, lr])
                                # last_value = results[key2][-1][plot_metric]
                                # key2 = tuple(list(key1v))
                                last_value = results[key2][-1][plot_metric]
                                vs.append(last_value)
                    # print(alg_method, update_method, m, cfg, vs)
                    mu, std = np.mean(vs), np.std(vs)
                    ys.append(mu)
                    ys_erros.append(1.96 * std / np.sqrt(len(vs)))  # std_error: 2*\sigma/sqrt(n)
            plot_data[f'{alg_method}-{update_method}'] = (xs, ys, ys_erros)
    return plot_data



def extract_data_proposed(results, cfg2, plot_metric='min_dist'):
    # plot_metric = 'min_dist'    # 'max_dist'    # min_loss, 'min_dist'
    ms = cfg2['m']
    ds = cfg2['d']
    ns = cfg2['n']
    ps = cfg2['p']
    plot_data = {}
    # print(f"m {cfg2['m'][0]}, r {cfg2['r'][0]}")
    print(f"n {cfg2['n'][0]}")

    alg_methods = cfg2['alg_method']
    update_methods = cfg2['update_method']
    xs = ds  # x-axis
    # file_name = 'ds'
    # x_label = 'Dim'
    # print(f'plot_metric: {plot_metric}, xs: {xs}, update_methods: {update_methods}')

    data_seeds = cfg2['data_seed']
    train_seeds = cfg2['train_seed']
    lrs = cfg2['lr']
    del cfg2['data_seed']
    del cfg2['train_seed']
    del cfg2['lr']
    del cfg2['alg_method']

    cfgs2 = list(product_dict(**cfg2))

    plot_data = {}
    for alg_method in alg_methods:
        for update_method in update_methods:
            ys = []
            ys_erros = []
            for m in xs:  # x_labels: different machines, y_label: metric
                for cfg in cfgs2:  # all results
                    key1 = cfg.keys()
                    key1v = cfg.values()
                    if cfg['d'] != m or cfg['update_method'] != update_method: continue
                    # if cfg['d'] != m or cfg['update_method'] != update_method or cfg[
                    #     'alg_method'] != alg_method: continue
                    # # THRE = THRE0 * cfg["noise_scale"]
                    success_rate = 0
                    vs = []
                    for d_i in data_seeds:
                        for t_i in train_seeds:
                            for lr in lrs:
                                key2 = tuple(list(key1v) + [d_i, t_i, lr])
                                # last_value = results[key2][-1][plot_metric]
                                # key2 = tuple(list(key1v))
                                last_value = results[key2][-1][plot_metric]
                                vs.append(last_value)
                    # print(alg_method, update_method, m, cfg, vs)
                    mu, std = np.mean(vs), np.std(vs)
                    ys.append(mu)
                    ys_erros.append(1.96 * std / np.sqrt(len(vs)))  # std_error: 2*\sigma/sqrt(n)
            plot_data[f'{alg_method}-{update_method}'] = (xs, ys, ys_erros)
    return plot_data

def plot_res(plot_data, key_algs=[('baseline-trimmed_mean', 'Baseline'), ], plot_metric='min_dist', out_dir='', pkl_file=''):
    import matplotlib;
    # matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend; instead, writes files
    from matplotlib import pyplot as plt
    if plot_metric == 'min_dist':
        # y_label = '$\\frac{1}{k}\sum||\\theta-\\theta^{*}||_2$'
        y_label = '$Avg. \\operatorname{dist}$'
    elif plot_metric == 'max_dist':
        y_label = '$max_{k}||\\theta-\\theta^{*}||_2$'
    else:
        y_label = plot_metric
    markers = ['*', '^', 'o', 'v']
    colors = ['g', 'b', 'purple', 'm']
    ecolors=['tab:brown', 'tab:red','tab:cyan', 'tab:olive']

    res = {}
    for _i, (key, alg_label) in enumerate(key_algs):
        xs, ys, ys_erros = plot_data[key]
        print(key, xs, ys, ys_erros)
        res[key] = (xs, ys, ys_erros)
        # plt.errorbar(xs, ys, yerr=ys_erros, marker=markers[_i], color=colors[_i], label=f"{alg_label}")
        # plt.plot(xs, ys, mc, label=f"{update_method}")
        plt.errorbar(xs, ys, yerr=ys_erros, marker=markers[_i], color=colors[_i],
                          capsize=3,  ecolor=ecolors[_i],
                          markersize=7, markerfacecolor='black',
                          label=f"{alg_label}", alpha=1)

    # p_{p}-m_{m}-n_100-Dim_{plot_metric}.pkl'
    # pkl_file = f"{out_dir}/n_100-{x_label}_{plot_metric}.pkl"
    print(pkl_file)
    with open(pkl_file, 'wb') as f:
        pickle.dump(res, f)
    x_label = 'd'
    # plt.title("r vs success rate")
    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")

    plt.legend(loc="upper left")

    # plt.xscale("log")
    # # plt.yscale("log")
    plt.tight_layout()

    # f = f"{out_dir}/n_100-{x_label}_{plot_metric}"
    f = f"{pkl_file}.png"
    plt.savefig(f, dpi=300)
    # plt.savefig(f"{f}.eps", format="eps", dpi=100)
    # plt.savefig(f"{f}.svg", format="svg", transparent=True)
    print(f)
    plt.show()
    plt.clf()
    plt.close()

    # # import ipdb; ipdb.set_trace()
    #
    # ##### plotting part #####
    # data1 = {}
    # data1['plot_data'] = plot_data
    # data1['cfg'] = cfg
    # print(plot_data)
    # with open("plot_data.pkl", 'wb') as f:
    #     pickle.dump(data1, f)


def merge_two_dicts(x, y):
    z = x.copy()
    # z.update(y)

    for k, v in y.items():
        if k in x.keys():
            raise KeyError(k)
        z[k] = v

    return z


def main_line():
    # results = {}
    # bs_file = os.path.join(in_dir, bs_file)
    # with open(bs_file, 'rb') as f:
    #     _results = pickle.load(f)
    # bs_results = merge_two_dicts(results, _results)

    in_dir = 'results_20230515'

    OUT_DIR = f'{in_dir}/paper_plots'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    cases = [
        ('K=2_baseline.pkl', 'K=2_our_algorithm.pkl', 2, 80),
        ('K=5_baseline.pkl', 'K=5_our_algorithm.pkl', 5, 200),
        ('K=10_baseline.pkl', 'K=10_our_algorithm.pkl', 10, 400),   # p:10, m:400
        ('K=15_baseline.pkl', 'K=15_our_algorithm.pkl', 15, 600)
    ]
    for plot_metric in ['min_dist']:  # ['min_dist', 'max_dist', 'min_loss']:
        for bs_file, prop_file, p, m, in cases:
            res_plot_pkl = f'{OUT_DIR}/{bs_file}_{plot_metric}.pkl'
            if os.path.exists(res_plot_pkl):
                print('loading results from {}'.format(res_plot_pkl))
                with open(res_plot_pkl, 'rb') as f:
                    plot_data = pickle.load(f)
            else:
                # 1. Get baseline result
                bs_file = os.path.join(in_dir, bs_file)
                with open(bs_file, 'rb') as f:
                    bs_results = pickle.load(f)
                cfg2 = get_CFG(p=[p], m=[m], alg_method=['baseline'], update_method=['trimmed_mean'])
                bs_plot_data = extract_data(bs_results, cfg2, plot_metric)

                # 2. Get proposed resutl
                prop_file = os.path.join(in_dir, prop_file)
                with open(prop_file, 'rb') as f:
                    prop_results = pickle.load(f)
                cfg2 = get_CFG(p=[p], m=[m], alg_method=['proposed'], update_method=['mean', 'median', 'trimmed_mean'])
                if prop_file.startswith('K=10_') or prop_file.startswith('K=15_'):
                    prop_plot_data = extract_data_proposed(prop_results, cfg2, plot_metric)  # for old results
                else:
                    prop_plot_data = extract_data(prop_results, cfg2, plot_metric)

                # 3. Combine two plot data and plot
                plot_data = merge_two_dicts(prop_plot_data, bs_plot_data)

            # key_algs = [('baseline-trimmed_mean', 'Baseline'),
            #             ('proposed-mean', 'FedAvg'), ('proposed-median', 'Median'),
            #             ('proposed-trimmed_mean', 'Trimmed-mean')]

            key_algs = [('baseline-trimmed_mean', 'Three-Stage'),
                        ('proposed-mean', 'FedAvg(IFCA)'),
                        ('proposed-median', 'Median(Alg1)'),
                        ('proposed-trimmed_mean', 'Trimmed-mean(Alg1)')]

            plot_res(plot_data, key_algs, plot_metric, out_dir=OUT_DIR, pkl_file =res_plot_pkl)
            # break

def main_bar_demo():
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ## the data
    N = 5
    menMeans = [18, 35, 30, 35, 27]
    menStd = [2, 3, 4, 1, 2]
    menMeans2 = [18-6, 35-2, 30-6, 35+6, 27-9]
    menStd2 = [2, 3, 4, 1, 2]
    menMeans3 = [18+2, 35+1, 30+3, 35-2, 27+1]
    menStd3 = [2, 3, 4, 1, 2]

    womenMeans = [25, 32, 34, 20, 25]
    womenStd = [3, 5, 2, 3, 3]

    ## necessary variables
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15  # the width of the bars

    ## the bars
    data = [
        (menMeans, menStd),
        (menMeans2, menStd2),
        (menMeans3, menStd3),
        (womenMeans, womenStd),
    ]
    rects = []
    markers = ['*', '^', 'o', 'v']
    colors = ['g', 'b', 'purple', 'm']
    ecolors = ['tab:brown', 'tab:red', 'tab:cyan', 'tab:olive']
    for i, (mean, std) in enumerate(data):
        rects_i = ax.bar(ind+i*width, mean, width,
                        color=colors[i],
                        yerr=std,
                        error_kw=dict(elinewidth=2, ecolor='red'))
        rects.append(rects_i)

    # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, 45)
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    xTickMarks = ['Group' + str(i) for i in range(1, 6)]
    ax.set_xticks(ind + width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    ax.legend(tuple(rects), ('Men', 'M1', 'M2',  'Women'))

    plt.show()


def main_bar():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ## the data

    data = []
    out_dir = 'results_20230515/paper_plots'
    # files = ['K=2_baseline.pkl_min_dist.pkl','K=5_baseline.pkl_min_dist.pkl','K=10_baseline.pkl_min_dist.pkl', 'K=15_baseline.pkl_min_dist.pkl']
    files = ['K=2_baseline.pkl_min_dist.pkl', 'K=5_baseline.pkl_min_dist.pkl',
             'K=15_baseline.pkl_min_dist.pkl']

    ## necessary variables
    N = len(files)
    ind = np.arange(N)  # the x locations for the groups
    # xTickMarks = ['K=2', 'K=5', 'K=10', 'K=15']
    xTickMarks = ['K=2', 'K=5', 'K=15']
    width = 0.18  # the width of the bars

    res = {}
    for i, file in enumerate(files):
        file = os.path.join(out_dir, file)
        print(file)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        xs = []
        means = []
        stds = []
        for k, vs in data.items():
            print(k, vs)
            xs, mean, std = vs[0], vs[1], vs[2]
            # if k not in res.keys():
            #     res[k] = [[], [], []]
            means.append(mean[-1])   # d=200 xs = [20, 50, 100, 200, 500]
            stds.append(std[-1])
        res[f'{i}_means'] = means
        res[f'{i}_stds'] = stds

    df = pd.DataFrame(res)
    ## the bars
    rects = []
    markers = ['*', '^', 'o', 'v']
    colors = ['g', 'b', 'purple', 'm']
    ecolors = ['tab:brown', 'tab:red', 'tab:cyan', 'tab:olive']
    for i in range(4):
        mean = df.iloc[i, ::2].values
        std = df.iloc[i, 1::2].values
        rects_i = ax.bar(ind+i*width, mean, width,
                        color=colors[i],
                        yerr=std,  capsize = 3,
                        error_kw=dict(elinewidth=2, ecolor='red', markerfacecolor = 'black', ))

        rects.append(rects_i)

    # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, 1.3)
    # y_label = '$\\frac{1}{k}\sum||\\theta-\\theta^{*}||_2$'
    y_label = '$Avg. \\operatorname{dist}$'
    ax.set_ylabel(y_label)
    # ax.set_title('Scores by group and gender')
    # xTickMarks = ['Group' + str(i) for i in range(1, 6)]
    ax.set_xticks(ind + width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=0, fontsize=10)
    ## add a legend
    # plt.legend(tuple(rects), ('Baseline', 'FedAvg', 'Median',  'Trimmed-mean'), bbox_to_anchor=(0.74,0.95))
    plt.legend(tuple(rects), ('Three-Stage', 'FedAvg(IFCA)', 'Median(Alg1)', 'Trimmed-mean(Alg1)'),
               bbox_to_anchor=(0.75, 0.95))
    # plt.legend(tuple(rects), ('Baseline', 'FedAvg', 'Median',  'Trimmed-mean'),loc='center left', bbox_to_anchor=(1,0.815), numpoints=1)
    plt.tight_layout()

    f = f'{out_dir}/bar.png'
    plt.savefig(f, dpi=300)

    plt.show()

if __name__ == '__main__':
    main_line()
    main_bar()
