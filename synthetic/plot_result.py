import os.path
import pickle

import copy
import numpy as np

from synthetic.process_runner import product_dict

OUT_DIR = 'plots'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Note that the last three parameters must be "data_seed, train_seed, and lr"
CFG = {
    "p": [5],  # number of distributions/clusters

    "m": [200],  # number of total machines (Normal + Byzantine)
    'alpha': [0.05],  # percent of Byzantine machines

    "n": [50],  # [50, 100],  # number of data points per each machine, [50, 100, 200, 400, 800]

    "d": [20, 50, 100, 200, 500],  # different data dimensions: [5, 25, 50, 100, 200]

    "noise_scale": [0.4472],  # standard deviation of noise/epsilon: sigma**2 = 0.2

    "r": [1.0],  # separation parameter for synthetic data generation

    "alg_method": ['baseline'], #   ['baseline', 'proposed'],

    'update_method': ['mean', 'median', 'trimmed_mean'], #gradient update methods for server
    'beta': [0.05],  # trimmed means parameters

    "data_seed": [v for v in range(0, 100, 2)],  # different seeds for data

    "train_seed": [0],  # different seeds for training

    'lr': [0.01],  # different learning rates
}

def plot_res(results, CFG, plot_metric):


    ###### processing ######

    cfg2 = copy.deepcopy(CFG)
    del cfg2['data_seed']
    del cfg2['train_seed']
    del cfg2['lr']
    # del cfg2['alg_method']

    cfgs2 = list(product_dict(**cfg2))

    plot_data = {}

    # print(f"m {cfg2['m'][0]}, r {cfg2['r'][0]}")
    print(f"n {cfg2['n'][0]}")
    # plot_metric = 'min_dist'    # 'max_dist'    # min_loss, 'min_dist'
    ms = CFG['m']
    ds = CFG['d']
    ns = CFG['n']
    ps = CFG['p']
    update_methods = CFG['update_method']
    xs = ds  # x-axis
    file_name = 'ds'
    x_label = 'd'
    print(f'plot_metric: {plot_metric}, xs: {xs}, update_methods: {update_methods}')

    plot_data = {}
    for update_method in update_methods:
        ys = []
        ys_erros = []
        for m in xs:  # x_labels: different machines, y_label: metric
            for cfg in cfgs2:  # all results
                key1 = cfg.keys()
                key1v = cfg.values()
                # if cfg['m'] != m or cfg['update_method'] != update_method: continue
                if cfg['d'] != m or cfg['update_method'] != update_method: continue

                # THRE = THRE0 * cfg["noise_scale"]

                success_rate = 0
                vs = []
                for d_i in CFG['data_seed']:
                    for t_i in CFG['train_seed']:
                        for lr in CFG['lr']:
                            key2 = tuple(list(key1v) + [d_i, t_i, lr])
                            last_value = results[key2][-1][plot_metric]
                            vs.append(last_value)
                print(update_method, m, cfg, vs)
                mu, std = np.mean(vs), np.std(vs)
                ys.append(mu)
                ys_erros.append(1.96 * std / np.sqrt(len(vs)))  # std_error: 2*\sigma/sqrt(n)
        plot_data[update_method] = (xs, ys, ys_erros)

    is_show = True
    if is_show:
        import matplotlib;
        # matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend; instead, writes files
        from matplotlib import pyplot as plt
        if plot_metric == 'min_dist':
            y_label = '$\\frac{1}{k}\sum||\\theta-\\theta^{*}||_2$'
        elif plot_metric == 'max_dist':
            y_label = '$max_{k}||\\theta-\\theta^{*}||_2$'
        else:
            y_label = plot_metric
        markers = ['*', '^', 'o']
        colors = ['g', 'b', 'purple']
        for _i, update_method in enumerate(update_methods):
            xs, ys, ys_erros = plot_data[update_method]
            print(update_method, xs, ys, ys_erros)
            plt.errorbar(xs, ys, yerr=ys_erros, marker=markers[_i], color=colors[_i], label=f"{update_method}")
            # plt.plot(xs, ys, mc, label=f"{update_method}")

        # plt.title("r vs success rate")
        plt.xlabel(f"{x_label}")
        plt.ylabel(f"{y_label}")

        plt.legend(loc="upper left")

        # plt.xscale("log")
        # # plt.yscale("log")
        plt.tight_layout()

        f = f"{OUT_DIR}/n_{ns[0]}-{x_label}_{plot_metric}"
        plt.savefig(f"{f}.png", dpi=100)
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

def main():
    results = {}
    for res_file in ['output1/results.pkl']:
        with open(res_file, 'rb') as f:
            _results = pickle.load(f)
        results = merge_two_dicts(results, _results)

    for plot_metric in ['min_dist', 'max_dist', 'min_loss']:
            plot_res(results, CFG, plot_metric)


if __name__ == '__main__':
    main()
