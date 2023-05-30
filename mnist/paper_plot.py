import os.path
import pickle
import shutil
import subprocess
import numpy as np
from functools import partial

print = partial(print, flush=True)

def main():

    cnt = 0
    alg_methods = ['proposed']      # 'baseline'
    update_methods = ["mean", "median", "trimmed_mean"]
    data_seeds = range(0, 100, 20)
    in_dir = 'alpha_01-beta_01' # 'small_set-alpha_01-beta_01'
    results_file = f'{in_dir}/results.pickle'

    if os.path.exists(results_file):
        os.remove(results_file)

    n_epochs = 100
    if not os.path.exists(results_file):
        results = {}
        for alg_method in alg_methods:
            results[alg_method] = {}
            for update_method in update_methods:
                vs = []
                for data_seed in data_seeds:
                    cnt += 1
                    _in_dir = f"{in_dir}/{alg_method}/{update_method}/{n_epochs}/{data_seed}"
                    _res_file = f'{_in_dir}/results.pickle'
                    print(cnt, _res_file)
                    with open(_res_file, 'rb') as f:
                        res = pickle.load(f)
                    # res[-1] # last epoch
                    # train.append()
                    # vs.append({'train':res[-1]['train'], 'test':res[-1]['test']})
                    vs.append(res)
                results[alg_method][update_method] = vs
        with open(results_file, 'wb') as f2:
            pickle.dump(results, f2)
    else:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

    # plot the results
    out_dir = os.path.join(os.path.dirname(results_file), 'plot_results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    markers = ['*', '^', 'o', '+']
    colors = ['g', 'b', 'purple', 'cyan']
    txt_results = []
    for train_method in ['train', 'test']:
        for plot_metric in ['acc', 'loss', 'infer_time', 'cl_acc']:
            for alg_method in alg_methods:
                for idx, update_method in enumerate(update_methods):
                    color = colors[idx]
                    marker = markers[idx]
                    n_epochs = len(results[alg_method][update_method][0])
                    xs = range(n_epochs)
                    ys = []
                    ys_errs = []
                    for i in range(n_epochs):
                        _xs = []
                        _ys = []
                        for j, data_seed in enumerate(data_seeds):
                            # if i == 0: continue
                            vs = results[alg_method][update_method][j][i]
                            # print(train_method, plot_metric, alg_method, update_method, j, i)
                            _ys.append(vs[train_method][plot_metric])
                        ys.append(np.mean(_ys))
                        ys_errs.append(1.96*np.std(_ys)/np.sqrt(len(data_seeds)))
                    print(train_method, plot_metric, alg_method, update_method, ys, ys_errs)
                    txt_results.append(' | '.join([train_method, plot_metric, alg_method, update_method, str(ys), str(ys_errs)]) + '\n\n')
                    # plot the results
                    is_show = True
                    if is_show:
                        import matplotlib;
                        # matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend; instead, writes files
                        from matplotlib import pyplot as plt
                        if plot_metric == 'loss':
                            # y_label = '$\\frac{1}{k}\sum||\\theta-\\theta^{*}||_2$'
                            y_label = 'Loss'
                        elif plot_metric == 'accuracy':
                            # y_label = '$max_{k}||\\theta-\\theta^{*}||_2$'
                            y_label = 'Accuracy'
                        else:
                            y_label = plot_metric
                        plt.errorbar(xs, ys, ys_errs,c=color, marker=marker, label=f"{train_method}-{update_method}-{plot_metric}")

            x_label = 'Epoch'
            plt.xlabel(f"{x_label}")
            plt.ylabel(f"{y_label}")
            plt.title(alg_method)
            plt.legend(loc="upper left")

            # plt.xscale("log")
            # # plt.yscale("log")
            plt.tight_layout()

            # f = f"{out_dir}/n_{ns[0]}-{x_label}_{plot_metric}"
            f = f'{out_dir}/{train_method}-{plot_metric}'
            plt.savefig(f"{f}.png", dpi=100)
            # plt.savefig(f"{f}.eps", format="eps", dpi=100)
            # plt.savefig(f"{f}.svg", format="svg", transparent=True)
            print(f)
            plt.show()
            plt.clf()
            plt.close()

    out_file = f'{out_dir}/results.txt'
    with open(out_file, 'w') as f:
        pickle.dump(txt_results, f)

    print(out_file)

    return cnt


if __name__ == '__main__':
    cnt = main()
    print(f'\n***total submitted jobs: {cnt}')
