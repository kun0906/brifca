"""
    module load anaconda3/2021.11
    conda activate py3104_ifca
    PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 run_all.py > 'output/log.txt' 2>&1 &

"""
import os
import json
import time
import pickle
import copy

import numpy as np

from process_runner import *

parser = argparse.ArgumentParser()
parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
                    action='store_true', help='force')
parser.add_argument("--max-procs", type=int, default=-1)  # -1 for debugging
parser.add_argument("--arr-size", type=int, default=-1)
parser.add_argument("--arr-index", type=int, default=-1)
parser.add_argument("-n", type=int, default=50)
parser.add_argument("-d", type=int, default=2)
parser.add_argument("--update_method", type=str, default='mean')
parser.add_argument("--alg_method", type=str, default='baseline')
args = parser.parse_args()

OUT_DIR = 'output_true_label'
OUT_DIR = 'output_60label'

def main(n=50):
    max_procs = 30

    is_debugging = True
    if is_debugging:
        # Note that the last three parameters must be "data_seed, train_seed, and lr"
        cfg = {
            "p": [15],  # number of distributions/clusters

            "m": [600],  # number of total machines (Normal + Byzantine)
            'alpha': [0.05],  # percent of Byzantine machines

            "n": [n],  # [50, 100],  # number of data points per each machine, [50, 100, 200, 400, 800]

            "d": [20, 50, 100, 200, 500],  # different data dimensions: [5, 25, 50, 100, 200]

            "noise_scale": [0.4472],  # standard deviation of noise/epsilon: sigma**2 = 0.2

            "r": [1.0],  # separation parameter for synthetic data generation

            "alg_method": ['baseline'], #   ['baseline', 'proposed'],

            'update_method': ['trimmed_mean'], #gradient update methods for server, 'mean', 'median',
            'beta': [0.05],  # trimmed means parameters

            "data_seed": [v for v in range(0, 100, 151)],  # different seeds for data

            "train_seed": [0],  # different seeds for training

            'lr': [0.01],  # different learning rates
        }
    # else:
    #     cfg = {
    #         "p": [5, 10],  # number of distributions/clusters
    #
    #         "m": [200],  # number of total machines (Normal + Byzantine)
    #         'alpha': [0.05, 0.1],  # percent of Byzantine machines
    #
    #         "n": [50, 100, 200],  # number of data points per each machine
    #
    #         "d": [20, 500, 1000],  # different data dimensions
    #
    #         "noise_scale": [0.4472],  # standard deviation of noise/epsilon
    #
    #         "r": [1.0],  # separation parameter for synthetic data generation
    #
    #         'update_method': ['mean', 'median', 'trimmed_mean'], #gradient update methods for server
    #         'beta': [0.05],  # trimmed means parameters
    #
    #         "data_seed": [v for v in range(0, 100, 20)],  # different seeds for data
    #
    #         "train_seed": [0],  # different seeds for training
    #
    #         'lr': [0.01],  # different learning rates
    #     }

    task = MyTask
    runner = MyProcessRunner(
        task,
        cfg,
        max_procs,
    )
    runner.setup()

    runner.run(force=args.force)
    runner.summarize(force=args.force)
    for plot_metric in ['min_dist', 'max_dist', 'min_loss']:
        # plot_metric = 'min_dist'  # 'max_dist'    # min_loss, 'min_dist'
        runner.plot_res(plot_metric=plot_metric)
    # runner.summarize(force=args.force)
    # runner.summarize(force=True)



class MyProcessRunner(ProcessRunner):
    def summarize_old(self, force=False):
        THRE0 = 0.6

        results_fname = f'{OUT_DIR}/results.pkl'
        if os.path.exists(results_fname) and not force:
            print('loading results from {}'.format(results_fname))
            with open(results_fname, 'rb') as f:
                results = pickle.load(f)
        else:
            print('start reading results...')
            results = {}

            t0 = time.time()

            eof_error_fnames = []

            for t_i, task in enumerate(self.tasks):
                cfg = task.cfg
                del cfg['project_dir']
                del cfg['dataset_dir']
                result_fname1 = task.procs[0].result_fname
                # print(cfg, result_fname)
                with open(result_fname1, 'rb') as f:
                    try:
                        res = pickle.load(f)
                        last_loss = res[-1]['min_dist']
                        key = tuple(cfg.values())
                        # print(key)

                        results[key] = last_loss

                    except EOFError as e:
                        eof_error_fnames.append(result_fname1)

                if t_i % 100 == 0:
                    print(f'reading {t_i}/{len(self.tasks)} done \r', end='')

            print('')

            print('removing eof_error files: ..')

            for fname in eof_error_fnames:
                print("remove:", fname)
                os.remove(fname)

            assert len(eof_error_fnames) == 0

            with open(results_fname, 'wb') as f:
                pickle.dump(results, f)

            t1 = time.time()
            print(f'reading and saving results done in {t1 - t0:.3f}sec')

        ###### processing ######

        cfg2 = copy.deepcopy(self.cfg)
        del cfg2['data_seed']
        del cfg2['train_seed']
        del cfg2['lr']

        cfgs2 = list(product_dict(**cfg2))

        plot_data = {}

        # print(f"m {cfg2['m'][0]}, r {cfg2['r'][0]}")
        print(f"n {cfg2['n'][0]}")

        for cfg in cfgs2:
            key1 = cfg.keys()
            key1v = cfg.values()

            THRE = THRE0 * cfg["noise_scale"]

            success_rate = 0
            min_values = []
            for d_i in self.cfg['data_seed']:
                is_success = False
                min_value = -1
                for t_i in self.cfg['train_seed']:
                    for lr in self.cfg['lr']:
                        key2 = tuple(list(key1v) + [d_i, t_i, lr])
                        last_value = results[key2]

                        if np.isnan(last_value):
                            continue

                        if min_value == -1:
                            min_value = last_value
                        elif min_value > last_value:
                            min_value = last_value
                        else:
                            pass

                min_values.append(min_value)

                is_success = (min_value < THRE)
                if is_success:
                    success_rate += 1

            success_rate /= len(self.cfg['data_seed'])

            print(cfg,
                  f' min {np.min(min_values):.5f} avg {np.mean(min_values):.5f} max {np.max(min_values):.5f} TH {THRE} sr {success_rate:.3f} ds {len(self.cfg["data_seed"])}')

            # print(cfg,
            #     f' min {np.min(min_values):.5f} avg {np.mean(min_values):.5f} max {np.max(min_values):.5f} ds {len(self.cfg["data_seed"])}')

            plot_data[(cfg['m'], cfg['n'])] = success_rate

        # import ipdb; ipdb.set_trace()

        ##### plotting part #####
        data1 = {}
        data1['plot_data'] = plot_data
        data1['cfg'] = self.cfg
        print(plot_data)
        with open("plot_data.pkl", 'wb') as f:
            pickle.dump(data1, f)

        # import matplotlib; matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend; instead, writes files
        # from matplotlib import pyplot as plt

        # noise_scales = self.cfg['noise_scale']
        # rs = self.cfg['r']

        # print("noise_scales")
        # print(noise_scales)
        # print("rs")
        # print(rs)

        # for noise_scale in noise_scales:
        #     success_rates = [ plot_data[(noise_scale,r)] for r in rs ]
        #     print(f"success_rates for noise noise_scale {noise_scale}")
        #     print(success_rates)
        #     plt.plot(rs, success_rates, label = f"noise_scale={noise_scale:.3f}")

        # # plt.title("r vs success rate")
        # plt.xlabel("r")
        # plt.ylabel("success rate")

        # plt.legend(loc="upper left")

        # plt.xscale("log")
        # # plt.yscale("log")

        # plt.savefig("plot.png", dpi=1000)
        # plt.savefig("plot.eps", dpi=1000)

    def summarize(self, force=False, plot_metric='min_dist'):
        THRE0 = 0.6

        results_fname = f'{OUT_DIR}/results.pkl'
        if os.path.exists(results_fname):
            os.rename(results_fname, results_fname + '-old.pkl')
        if os.path.exists(results_fname) and not force:
            print('loading results from {}'.format(results_fname))
            with open(results_fname, 'rb') as f:
                results = pickle.load(f)
        else:
            print('start reading results...')
            results = {}

            t0 = time.time()

            eof_error_fnames = []

            for t_i, task in enumerate(self.tasks):
                cfg = task.cfg
                if 'project_dir' in cfg.keys():
                    del cfg['project_dir']
                if 'dataset_dir' in cfg.keys():
                    del cfg['dataset_dir']
                result_fname1 = task.procs[0].result_fname
                # print(cfg, result_fname)
                with open(result_fname1, 'rb') as f:
                    try:
                        res = pickle.load(f)
                        # last_loss = res[-1]['min_dist']
                        key = tuple(cfg.values())
                        # print(key)

                        # results[key] = last_loss
                        results[key] = res

                    except EOFError as e:
                        eof_error_fnames.append(result_fname1)

                if t_i % 100 == 0:
                    print(f'reading {t_i}/{len(self.tasks)} done \r', end='')

            print('')

            print('removing eof_error files: ..')

            for fname in eof_error_fnames:
                print("remove:", fname)
                os.remove(fname)

            assert len(eof_error_fnames) == 0

            with open(results_fname, 'wb') as f:
                pickle.dump(results, f)

            t1 = time.time()
            print(f'reading and saving results done in {t1 - t0:.3f}sec')

    def plot_res(self, plot_metric):

        results_fname = f'{OUT_DIR}/results.pkl'
        with open(results_fname, 'rb') as f:
            results = pickle.load(f)

        ###### processing ######

        cfg2 = copy.deepcopy(self.cfg)
        del cfg2['data_seed']
        del cfg2['train_seed']
        del cfg2['lr']

        cfgs2 = list(product_dict(**cfg2))

        plot_data = {}

        # print(f"m {cfg2['m'][0]}, r {cfg2['r'][0]}")
        print(f"n {cfg2['n'][0]}")
        # plot_metric = 'min_dist'    # 'max_dist'    # min_loss, 'min_dist'
        ms = self.cfg['m']
        ds = self.cfg['d']
        ns = self.cfg['n']
        ps = self.cfg['p']
        update_methods = self.cfg['update_method']
        xs = ds  # x-axis
        file_name = 'ds'
        x_label = 'd'
        print(f'plot_metric: {plot_metric}, xs: {xs}, update_methods: {update_methods}')

        plot_data = {}
        for update_method in update_methods:
            ys = []
            ys_erros = []
            for m in xs:    # x_labels: different machines, y_label: metric
                for cfg in cfgs2: # all results
                    key1 = cfg.keys()
                    key1v = cfg.values()
                    # if cfg['m'] != m or cfg['update_method'] != update_method: continue
                    if cfg['d'] != m or cfg['update_method'] != update_method: continue

                    # THRE = THRE0 * cfg["noise_scale"]

                    success_rate = 0
                    vs = []
                    for d_i in self.cfg['data_seed']:
                        for t_i in self.cfg['train_seed']:
                            for lr in self.cfg['lr']:
                                key2 = tuple(list(key1v) + [d_i, t_i, lr])
                                last_value = results[key2][-1][plot_metric]
                                vs.append(last_value)
                    print(update_method, m, cfg, vs)
                    mu, std = np.mean(vs), np.std(vs)
                    ys.append(mu)
                    ys_erros.append(1.96*std/np.sqrt(len(vs)))  # std_error: 2*\sigma/sqrt(n)
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
                plt.errorbar(xs, ys, yerr=ys_erros, marker=markers[_i], color = colors[_i], label=f"{update_method}")
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
        # data1['cfg'] = self.cfg
        # print(plot_data)
        # with open("plot_data.pkl", 'wb') as f:
        #     pickle.dump(data1, f)



class MyTask(PRTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        env0 = {"OMP_NUM_THREADS": str(1)}

        project_dir = os.path.join(OUT_DIR, dict_string(cfg))

        cfg["project_dir"] = project_dir
        cfg["dataset_dir"] = project_dir

        proc1 = PRProcess(
            command=[
                "python", "-u", "gen_data_and_train_cluster.py",
                "--config-override", json.dumps(cfg)
            ],
            output_dir=project_dir,
            result_fname='results.pickle',
            cleanup_fnames=[],
            env=env0,
            stdout_prefix="gen_data_and_train_cluster"
        )

        self.procs = [proc1]


if __name__ == '__main__':
    start_time = time.time()
    print(args)
    for n in [100]: #[50, 100, 500]:
        main(n)
    duration = (time.time() - start_time)
    print("---run.py Ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
