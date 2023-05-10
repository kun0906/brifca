import argparse
import json
import os
import time

import torch
import numpy as np

from util import *

parser = argparse.ArgumentParser()
parser.add_argument("--project-dir", type=str, default="output")
parser.add_argument("--data-seed", type=int, default=0)
parser.add_argument("--config-override", type=str, default="")
args = parser.parse_args()


def main():
    config = get_config()
    print("config:", config)

    exp = DatasetGenerate(config)
    exp.setup()
    dataset = exp.generate_dataset()
    exp.save()

    # exp.check_dataset() # for debugging
    pass


def get_config():
    # read config json and update the sysarg
    with open("config.json", "r") as read_file:
        config = json.load(read_file)

    args_dict = vars(args)
    config.update(args_dict)

    if config["config_override"] == "":
        del config['config_override']
    else:
        # print(config['config_override'])
        config_override = json.loads(config['config_override'])
        del config['config_override']
        config.update(config_override)

    return config


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

class DatasetGenerate(object):
    def __init__(self, config, seed=0):
        self.seed = config['data_seed']
        self.config = config

        assert self.config['m'] % self.config['p'] == 0

    def setup(self):
        # print('seeding', self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        self.dataset_dir = os.path.join(self.config['project_dir'])
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.dataset_fname = os.path.join(self.dataset_dir, 'dataset.pth')

        # param settings
        # p even -> loc: [-3, -1, 1, 3] p = 4
        # p odd -> loc: [-6, -4, -2, 0, 2, 4, 6] p = 7
        p = int(self.config['p'])
        self.param_settings = [-p + 1 + 2 * i for i in range(p)]

    def generate_dataset(self):
        p = self.config['p']
        d = self.config['d']
        m = self.config['m']    # Number of total machines = m_n (normal) + m_b (byzantine)
        n = self.config['n']    # Number of data points in each machine
        # r = separation parameter for synthetic data generation

        dataset = {}
        dataset['config'] = self.config

        # generate parameter set for each cluster/group
        params = []
        for p_i in range(p):
            # loc = self.param_settings[p_i]
            # generate d points from a binomial(n=1, p=0.5, size=d),
            # where each sample is equal to the number of successes over the n trials.
            # r: separation parameter for synthetic data generation
            while True:
                # n = 1 is Bernoulli distribution
                param = torch.tensor(np.random.binomial(n=1, p=0.5, size=(d)).astype(np.float32)) * self.config['r']
                if torch.linalg.norm(param, 2) > 0: break
            param = param/torch.linalg.norm(param, 2)  # normalize the vector by l2 norm
            # param = np.zeros((d))
            # param[p_i] = 10
            # param = torch.tensor(param.astype(np.float32)) # for debugging
            params.append(param)  # generate data parameters for each distribution/cluster
        print('Normal params(weights):', params)
        dataset['params'] = params
        dataset['data'] = []

        # generate dataset for each normal machine
        m_n = self.config['m_n']
        cluster_assignment = [m_i // int(np.ceil(m_n/p)) for m_i in range(m_n)]  # generate label for each machine
        dataset['cluster_assignment'] = cluster_assignment

        for m_i in range(m_n):
            p_i = cluster_assignment[m_i]  # the ith distribution's parameters
            loc = p_i * 10
            data_X = random_normal_tensor(loc=0, scale=1,
                size=(n, d))  # generate standard normal distribution N(0, 1) with size n and dim=d.
            data_y = data_X @ params[p_i]  # mixture of gaussians, X*theta

            noise_y = random_normal_tensor(size=(n), scale=self.config['noise_scale'])
            data_y = data_y + noise_y  # add noise to y, X*theta + eps_i
            data_y_label = [f'normal_{p_i}'] * len(data_y)

            dataset['data'].append((data_X, data_y, data_y_label))

        # plot_data(dataset['data'], cluster_assignment)
        ###################################################
        # generate data for each Byzantine machine
        params_b = []
        # p_b = 1  # only one distribution for all noise distribution
        m_b = self.config['m_b']
        p_b = m_b  # each Byzantine machine has one distribution
        for p_i in range(p_b):
            # loc = self.param_settings[p_i]
            while True:
                param = torch.tensor(np.random.binomial(1, 0.5, size=(d)).astype(np.float32)) * self.config['r']
                if torch.linalg.norm(param, 2) > 0: break
            param = 3 * param / torch.linalg.norm(param, 2)  # l2 norm = 3
            params_b.append(param)
        print('Byzantine params(weights):', params_b)
        # generate dataset for each Byzantine machine
        cluster_assignment_b = [m_i // int(np.ceil(m_b // p_b)) + p for m_i in
                                range(m_b)]  # generate label for each Byzantine machine
        dataset['cluster_assignment'] = cluster_assignment + cluster_assignment_b

        for m_i in range(m_b):
            p_i = cluster_assignment_b[m_i]-p  # the ith distribution's parameters
            loc = (p_i + 1) * 5
            data_X = random_normal_tensor(loc=0, scale=1,
                size=(n, d))  # generate standard normal distribution N(0, 1) with size n and dim=d.
            data_y = data_X @ params_b[p_i]  # mixture of gaussians

            noise_y = random_normal_tensor(size=(n), loc=0, scale=self.config['noise_scale'])
            data_y = data_y + noise_y  # add noise to y
            data_y_label = [f'Byzantine_{p_i}'] * len(data_y)

            dataset['data'].append((data_X, data_y, data_y_label))

        plot_data(dataset['data'], dataset['cluster_assignment'])

        self.dataset = dataset
        return dataset

    def save(self):
        torch.save(self.dataset, self.dataset_fname)

        from pathlib import Path
        Path(os.path.join(self.config['project_dir'], 'result_data.txt')).touch()

    def check_dataset(self):
        dataset = torch.load(self.dataset_fname)


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---generate_synthetic_cluster Ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
