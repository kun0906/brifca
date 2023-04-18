import argparse
import json
import os
import time
import itertools
import pickle

import torch
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn_extra.cluster import KMedoids

from util import *

parser = argparse.ArgumentParser()
parser.add_argument("--project-dir", type=str, default="output")
parser.add_argument("--dataset-dir", type=str, default="output")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--train-seed", type=int, default=0)
parser.add_argument("--config-override", type=str, default="")
args = parser.parse_args()

# LR_DECAY = True
LR_DECAY = False


def main():
    config = get_config()
    print("config:", config)

    exp = TrainCluster(config)
    exp.setup()
    exp.run()

    # exp.check_dataset() # for debugging
    # exp.cleanup()
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
        print(config['config_override'])
        config_override = json.loads(config['config_override'])
        del config['config_override']
        config.update(config_override)

    return config


class TrainCluster(object):
    def __init__(self, config):
        self.seed = config['train_seed']
        self.random_state = self.seed
        self.config = config

        assert self.config['m'] % self.config['p'] == 0

    def setup(self, dataset=None):
        # print('seeding', self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        self.output_dir = os.path.join(self.config['project_dir'], 'results.pickle')
        self.dataset_fname = os.path.join(self.config['dataset_dir'], 'dataset.pth')

        if dataset != None:
            self.dataset = dataset
        else:
            self.dataset = torch.load(self.dataset_fname)

        p = self.config['p']
        d = self.config['d']
        m = self.config['m']
        n = self.config['n']

        self.models = [SimpleLinear(input_size=d) for p_i in
                       range(p)]  # p models with p different params of dimension(1,d)
        self.criterion = torch.nn.MSELoss()

        self.epoch = None

        self.lr_decay_info = None

    def run(self):
        num_epochs = self.config['num_epochs']
        lr = self.config['lr']

        results = []
        for epoch in range(num_epochs):
            if epoch == 0:
                init_method = self.config['init_method']
                if init_method == 'server_random':
                    self.initialize_weights()  # server do the random initialization first
                elif init_method == 'client_kmeans++' or init_method == 'client_random':
                    self.client_init_first(init_method=init_method)
                else:
                    raise NotImplementedError(f'{init_method}')
                # self.warm_start()
            else:
                self.epoch = epoch
                result = self.train(lr=lr)

                result['epoch'] = epoch
                results.append(result)

                print(
                    f" epoch {self.epoch} min_loss {result['min_loss']:3f} min_dist {result['min_dist']:3f} lr {lr:.5f}")
                # print(f"      min_losses {result['min_losses']}")
                print('cluster_size:', result["cluster_assignment_ct"], ' cluster_label:', result["closest_cluster"])

                if LR_DECAY and self.determine_lr_decay(result):
                    # lr = lr * 0.1
                    lr = lr * 0.8

                if LR_DECAY and lr < 0.0001:
                    print('break due to lr < 0.0001')
                    break

        with open(self.output_dir, 'wb') as outfile:
            pickle.dump(results, outfile)
            print(f'result written at {self.output_dir}')

    def cleanup(self):
        os.remove(self.dataset_fname)

    def determine_lr_decay(self, result):

        if self.lr_decay_info == None:
            self.lr_decay_info = {}
            dd = self.lr_decay_info
            dd['ct'] = 0
            dd['loss'] = -1

        dd = self.lr_decay_info
        if dd['loss'] == -1:
            dd['loss'] = result['min_loss']
        else:
            if dd['loss'] - 1.0 > result['min_loss']:
                # still converging
                dd['ct'] = 0
                dd['loss'] = result['min_loss']
            else:
                # maybe converged
                dd['ct'] += 1

        if dd['ct'] > 5:
            dd['loss'] = result['min_loss']
            dd['ct'] = 0
            return True
        else:
            return False

    def train(self, lr):
        p = self.config['p']  # number of clusters/distributions
        d = self.config['d']  # number of dimension of the dataset
        m = self.config['m']  # number of normal machines
        n = self.config['n']  # number of data points of each machine
        m_b = self.config['m_b']  # number of byzantine machines

        result = {}

        # calc loss and grad
        losses = {}
        grads = {}

        flag_machines = np.asarray(
            [True if self.dataset['data'][m_i][-1][0].startswith('normal') else False for m_i in range(m + m_b)])

        # for each machine, compute the losses and grads of all distributions/clusters
        for m_i in range(m + m_b):
            for p_i in range(p):
                (X, y, y_label) = self.dataset['data'][m_i]
                if flag_machines[m_i]:  # y_label[0].startswith('normal'):  # Normal loss
                    loss_val, grad = calculate_loss_grad(self.models[p_i], self.criterion, X, y)
                else:  # Byzantine loss
                    loss_val, grad = 0, 0
                losses[(m_i, p_i)] = loss_val
                grads[(m_i, p_i)] = grad

        # calculate scores
        scores = {}
        for m_i in range(m + m_b):
            if not flag_machines[m_i]: continue  # only for normal machines

            machine_losses = [losses[(m_i, p_i)] for p_i in range(p)]  # for each machine, find all "p" losses

            if self.config['score'] == 'set':
                min_p_i = np.argmin(
                    machine_losses)  # for each machine, find the distribution that has the minimal loss.
                for p_i in range(p):
                    if p_i == min_p_i:
                        scores[(m_i, p_i)] = 1  # assign the minimal distribution to 1.
                    else:
                        scores[(m_i, p_i)] = 0

            elif self.config['score'] == 'em':

                from scipy.special import softmax
                softmaxed_loss = softmax(machine_losses)
                for p_i in range(p):
                    scores[(m_i, p_i)] = softmaxed_loss[p_i]

            else:
                assert self.config['score'] in ['set', 'em']

        # apply gradient update at server (here we only update normal machines)
        for p_i in range(p):
            cluster_scores = [scores[(m_i, p_i)] for m_i in range(m + m_b) if flag_machines[m_i]]
            cluster_grads = [grads[(m_i, p_i)] for m_i in range(m + m_b) if flag_machines[m_i]]

            self.models[p_i].zero_grad()
            weight = self.models[p_i].weight()
            # average the grads
            # tmp = gradient_update(cluster_scores, cluster_grads)  # should use "median"?
            tmp = gradient_update_median(cluster_scores, cluster_grads)  # should use "median"?
            weight.data -= lr * tmp

        # evaluate min_losses. for each machine, find the minimal loss and corresponding label
        min_losses = []
        cluster_assignment = []
        for m_i in range(m + m_b):
            if not flag_machines[m_i]: continue  # only normal machines
            machine_losses = [losses[(m_i, p_i)] for p_i in range(p)]
            min_loss = np.min(machine_losses)  # for each machine, find the minimal loss
            min_losses.append(min_loss)

            machine_scores = [scores[(m_i, p_i)] for p_i in range(p)]
            assign = np.argmax(machine_scores)  # find the distribution/cluster label

            cluster_assignment.append(assign)

        result["min_loss"] = np.mean(min_losses)  # average the minimal losses
        result["min_losses"] = min_losses

        cluster_assignment_ct = [0 for p_i in range(p)]
        for m_i in range(m + m_b):
            if flag_machines[m_i]:  # normal machine
                cluster_assignment_ct[cluster_assignment[m_i]] += 1  # for each cluster/distribution, compute the size.
            else:  # random assign the machine to one cluster, should I control the random state.
                r = np.random.RandomState(seed=m_i)
                idx = r.choice(range(p), size=1, replace=True)[0]  # random choose value from [0, p]
                cluster_assignment_ct[idx] += 1

        result["cluster_assignment_ct"] = cluster_assignment_ct

        closest_cluster = [-1 for _ in range(p)]
        min_dists = []
        for p_i in range(p):
            weight = self.models[p_i].weight()
            distances = []
            for p_j in range(p):
                param_ans = self.dataset['params'][p_j]  # ground-truth
                distances.append(torch.norm(weight.data - param_ans, 2))
            closest_cluster[p_i] = np.argmin(
                distances)  # for each distribution/cluster, find the label with the minimal L2**2 distance
            min_dist = np.min(distances)
            min_dists.append(min_dist)

        result["min_dist"] = np.mean(min_dists)  # average distances
        result["min_dists"] = min_dists

        result["closest_cluster"] = closest_cluster

        return result

    def initialize_weights(self):
        p = self.config['p']
        random_number = np.random.normal()  # dummy

        for p_i in range(p):
            weight = self.models[p_i].weight()
            d = weight.shape[1]
            param = torch.tensor(np.random.binomial(1, 0.5, size=(1, d)).astype(np.float32)) * 1.0
            weight.data = param

    def warm_start(self):
        # use the ground-truth to initialize, i.e., set the initialization to values that are close to ground-truth.
        noise_scale = 5.0

        p = self.config['p']

        for p_i in range(p):
            weight = self.models[p_i].weight()
            param_ans = self.dataset['params'][p_i]

            noise = random_normal_tensor(size=weight.data.shape, loc=0, scale=noise_scale)
            weight.data = param_ans + noise

        # compare the distance the distance to verify
        closest_cluster = [-1 for _ in range(p)]
        for p_i in range(p):
            weight = self.models[p_i].weight()
            distances = []
            for p_j in range(p):
                param_ans = self.dataset['params'][p_j]
                distances.append(torch.norm(weight.data - param_ans, 2))
            closest_cluster[p_i] = np.argmin(distances)

        assert closest_cluster == list(range(p)), f"closest_cluster {closest_cluster} != list(range(p))"

    def client_init_first(self, init_method='client_kmeans++'):

        p = self.config['p']  # number of clusters/distributions
        d = self.config['d']  # number of dimension of the dataset
        m = self.config['m']  # number of normal machines
        n = self.config['n']  # number of data points of each machine
        m_b = self.config['m_b']  # number of byzantine machines

        # flag_machines = np.asarray(
        #     [True if self.dataset['data'][m_i][-1][0].startswith('normal') else False for m_i in range(m + m_b)])

        # for each machine (client), find p centroids
        params = []
        for m_i in range(m + m_b):  # use all the machines' data
            (X, y, y_label) = self.dataset['data'][m_i]
            if 'kmeans++' in init_method:
                centroids, indices = kmeans_plusplus(X.detach().cpu().numpy(), p, x_squared_norms=None,
                                                     random_state=self.random_state)
            elif 'random' in init_method:
                r = np.random.RandomState(seed=self.random_state)
                indices = r.choice(range(len(y)), size=p, replace=False)
                centroids = X[indices]
            else:
                raise NotImplementedError(init_method)
            params.extend(centroids)
        params = np.asarray(params)

        # the server find the final centroids/params by kmeans++
        km = KMedoids(n_clusters=p, metric='euclidean', method='alternate', init='k-medoids++', max_iter=300,
                      random_state=self.random_state)
        km.fit(params)
        centroids = km.cluster_centers_

        # assign the initialization: maybe mismatch with ground-truth
        for p_i in range(p):
            weight = self.models[p_i].weight()
            # d = weight.shape[1]
            # param = torch.tensor(np.random.binomial(1, 0.5, size=(1, d)).astype(np.float32)) * 1.0
            param = torch.tensor(centroids[p_i].astype(np.float32)) * 1.0
            weight.data = param


###   ####  ###

def calculate_loss_grad(model, criterion, X, y):
    y_target = model(X)
    loss = criterion(y, y_target)
    model.zero_grad()
    loss.backward()

    loss_value = loss.item()
    weight = model.weight()
    d_weight = weight.grad.clone()

    return loss_value, d_weight


def gradient_update(scores, grads):
    m = len(grads)
    tmp = torch.zeros_like(grads[0])
    for m_i in range(m):
        tmp += scores[m_i] * grads[m_i]
    tmp /= m

    return tmp


def gradient_update_median(scores, grads):
    scores = np.asarray(scores)
    grads = torch.stack(grads)
    mask = (scores == 1)
    if sum(mask) == 1:
        tmp = grads[mask]
    elif sum(mask) > 1:
        tmp, indices = torch.median(grads[mask], dim=0)
    else:
        tmp = torch.zeros_like(grads[0])
    return tmp


class SimpleLinear(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, 1, bias=False)

    def weight(self):
        return self.linear1.weight

    def forward(self, x):
        return self.linear1(x).view(-1)  # 1 dim


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
