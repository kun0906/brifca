import argparse
import collections
import copy
import json
import os
import time
import itertools
import pickle

import matplotlib.pyplot as plt
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


def plot_data(results, f='.png', self=None):
    n = len(results)
    X = [i for i in range(n)]
    y = [vs['min_loss'] for vs in results]
    plt.plot(X, y, '-*', color='purple', label='min_loss')

    y = [vs['min_dist'] for vs in results]
    plt.plot(X, y, '-^', color='b', label='min_dist')

    y = [vs['max_dist'] for vs in results]
    plt.plot(X, y, '-o', color='g', label='max_dist')

    plt.xlabel('Epoch')
    plt.legend()
    update_method = self.config['update_method']
    plt.title(update_method)

    plt.tight_layout()
    plt.savefig(f, dpi=100)
    # plt.show()
    plt.clf()


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
        self.criterion = torch.nn.MSELoss()  # mean squared loss

        self.epoch = None

        self.lr_decay_info = None

    def run(self):
        num_epochs = self.config['num_epochs']
        lr = self.config['lr']
        # self.initialize_weights()

        results = []
        for epoch in range(num_epochs):
            if epoch == 0:
                init_method = self.config['init_method']
                if init_method == 'server_omniscient':
                    self.initialize_weights()  # server do the initialization that close to ground-truth
                # elif init_method == 'client_kmeans++' or init_method == 'client_random':
                #     self.client_init_first(init_method=init_method)
                else:
                    raise NotImplementedError(f'{init_method}')
                # self.warm_start()
            else:
                self.epoch = epoch

                # training phase
                result = self.train(lr=lr)

                result['epoch'] = epoch
                results.append(result)

                print(
                    f"{self.epoch}-th epoch, min_loss(y-y^*):{result['min_loss']:3f}, "
                    f"mean_l2_dist(theta-theta^*):{result['min_dist']:3f}, "
                    f"max_l2_dist(theta-theta^*) {result['max_dist']:3f}, "
                    f"lr: {lr:.5f}")
                # print(f"      min_losses {result['min_losses']}")
                print('\tcluster_size:', result["cluster_assignment_ct"],
                      ', pred_true:', result["cluster_assignment_name_ct"],
                      '\n\tcluster_label(closest_cluster):', result["closest_cluster"],
                      ', weights:', result['weights'])

                if LR_DECAY and self.determine_lr_decay(result):
                    # lr = lr * 0.1
                    lr = lr * 0.8

                if LR_DECAY and lr < 0.0001:
                    print('break due to lr < 0.0001')
                    break

        f = os.path.join(self.config['dataset_dir'], 'loss.png')
        plot_data(results, f, self)

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
        m = self.config['m']  # number of total machines
        n = self.config['n']  # number of data points of each machine
        m_n = self.config['m_n']  # number of normal machines
        m_b = self.config['m_b']  # number of byzantine machines


        result = {}

        # calc loss and grad
        losses = {}
        grads = {}

        flag_machines = np.asarray(
            [True if self.dataset['data'][m_i][-1][0].startswith('normal') else False for m_i in range(m_n + m_b)])

        # Clients:
        # for each machine, compute the losses and grads of all distributions/clusters (we need the labels)
        for m_i in range(m_n + m_b):
            for p_i in range(p):
                (X, y, y_label) = self.dataset['data'][m_i]
                if flag_machines[m_i]:  # y_label[0].startswith('normal'):  # Normal loss
                    loss_val, grad = calculate_loss_grad(self.models[p_i], self.criterion, X, y) # just compute the grads, no updates
                    # loss_val, grad = calculate_loss_grad(copy.deepcopy(self.models[p_i]), self.criterion, X, y)
                else:  # Byzantine loss
                    tmp_model = copy.deepcopy(self.models[p_i])
                    ws = tmp_model.weight()
                    ws.data = 3 * ws.data
                    loss_val, grad = calculate_loss_grad(tmp_model, self.criterion, X, y)
                    # loss_val, grad = calculate_loss_grad(copy.deepcopy(self.models[p_i]), self.criterion, X, y)
                    # loss_val, grad = 0, 0
                losses[(m_i, p_i)] = loss_val
                grads[(m_i, p_i)] = grad
                # print(m_i, p_i, self.models[p_i].weight())

        # calculate scores
        scores = {}
        for m_i in range(m_n + m_b):
            # if not flag_machines[m_i]: continue  # only for normal machines

            machine_losses = [losses[(m_i, p_i)] for p_i in range(p)]  # for each machine, find all "p" losses

            if self.config['score'] == 'set':
                min_p_i = np.argmin(
                    machine_losses)  # for each machine, find the distribution that has the minimal loss.
                for p_i in range(p):
                    if p_i == min_p_i:  # if there are more than p that has the same min_p_i, does it effect the final results?
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

        # Server:
        # apply gradient update at server (here we only update normal machines)
        weights = []
        for p_i in range(p):
            # find which machine is assigned to p_i. If scores([m_i, p_i]) == 1, m_i is assigned to p_i;
            # otherwise, m_i is not.
            cluster_scores = [scores[(m_i, p_i)] for m_i in range(m_n + m_b)]
            if sum(cluster_scores) == 0:    # Kun: if there is no machine assigned to p_i, we randomly select a machine.
                # print(cluster_grads)
                _idx = np.random.choice(m_n+m_b, size=1)[0]
                for j in range(p):
                    if j != p_i: scores[(_idx, j)] = 0  # assign the other value to 0 for this machine.
                scores[(_idx, p_i)] = 1 #p_i
                cluster_scores = [scores[(_idx, p_i)]]
                cluster_grads = [grads[(_idx, p_i)]]
                print(p_i, _idx, cluster_scores, cluster_grads)
            else:
                cluster_grads = [grads[(m_i, p_i)] for m_i in range(m_n + m_b)]
            # cluster_scores = [scores[(m_i, p_i)] for m_i in range(m_n + m_b) if flag_machines[m_i]]
            # cluster_grads = [grads[(m_i, p_i)] for m_i in range(m_n + m_b) if flag_machines[m_i]]

            self.models[p_i].zero_grad()
            weight = self.models[p_i].weight()

            update_method = self.config['update_method']
            if update_method == 'mean': #coordinate_wise_mean
                # average the grads
                tmp = gradient_update(cluster_scores, cluster_grads)
            elif update_method == 'median': # coordinate_wise_median
                tmp = gradient_update_median(cluster_scores, cluster_grads)  # should use "median"?
            elif update_method == 'trimmed_mean':   # trimmed_mean
                beta = self.config['beta']  # parameter for trimmed mean
                tmp = gradient_trimmed_mean(cluster_scores, cluster_grads, beta)
            else:
                raise NotImplementedError(f'{update_method}')

            weight.data -= lr * tmp
            # normalize the weight
            # weight.data = weight.data/torch.linalg.norm(weight.data, 2)
            weights.append(weight.detach().numpy())

        # Client:
        # evaluate min_losses. for each machine, find the minimal loss and corresponding label
        min_losses = []
        cluster_assignment = []
        for m_i in range(m_n + m_b):
            # if not flag_machines[m_i]: continue  # only normal machines
            machine_losses = [losses[(m_i, p_i)] for p_i in range(p)]
            min_loss = np.min(machine_losses)  # for each machine, find the minimal loss
            min_losses.append(min_loss)

            machine_scores = [scores[(m_i, p_i)] for p_i in range(p)]
            assign = np.argmax(machine_scores)  # find the distribution/cluster label

            cluster_assignment.append(assign)

        result["min_loss"] = np.mean(min_losses)  # average the minimal losses
        result["min_losses"] = min_losses
        print(collections.Counter(cluster_assignment))
        cluster_assignment_ct = [0 for p_i in range(p)]
        cluster_assignment_name_ct = {}
        for m_i in range(m_n + m_b):
            cluster_assignment_ct[cluster_assignment[m_i]] += 1  # for each cluster/distribution, compute the size.
            (X, y, y_label) = self.dataset['data'][m_i]
            key = cluster_assignment[m_i]  # each machine has only one unique label
            if key not in cluster_assignment_name_ct.keys():
                cluster_assignment_name_ct[key]=[y_label[0]]
            else:
                cluster_assignment_name_ct[key].append(y_label[0])
            # if flag_machines[m_i]:  # normal machine
            #     cluster_assignment_ct[cluster_assignment[m_i]] += 1  # for each cluster/distribution, compute the size.
            # else:  # random assign the machine to one cluster, should I control the random state?
            #     r = np.random.RandomState(seed=m_i)
            #     idx = r.choice(range(p), size=1, replace=True)[0]  # random choose value from [0, p)
            #     cluster_assignment_ct[idx] += 1

        result["cluster_assignment_ct"] = cluster_assignment_ct
        result["cluster_assignment_name_ct"] = [f'{k}:{collections.Counter(vs)}' for k,vs in cluster_assignment_name_ct.items()]
        # record the total loss for the updated weights and ground-truth for plotting or analyzing.
        closest_cluster = [-1 for _ in range(p)]
        min_dists = []
        max_dists = []
        for p_i in range(p):
            weight = self.models[p_i].weight()
            distances = []
            for p_j in range(p):
                param_ans = self.dataset['params'][p_j]  # ground-truth
                distances.append(torch.norm(weight.data - param_ans, 2))  # l2 norm: not squared l2 distance
            closest_cluster[p_i] = np.argmin(
                distances)  # for each distribution/cluster, find the label with the minimal L2**2 distance
            min_dist = np.min(distances)
            min_dists.append(min_dist)

        result["min_dist"] = np.mean(min_dists)  # average distances
        result["min_dists"] = min_dists
        result["max_dist"] = np.max(min_dists)  # max distances

        result["closest_cluster"] = closest_cluster
        result['weights'] = weights

        return result

    def initialize_weights(self):
        p = self.config['p']  # number of distributions
        # random_number = np.random.normal()  # dummy

        for p_i in range(p):
            weight = self.models[p_i].weight()  # uniform(-, +)
            d = weight.shape[1]
            # initialize the weights are close to ground-truth
            # param = torch.tensor(np.random.binomial(1, 0.5, size=(1, d)).astype(np.float32)) * 1.0    # wrong initalization
            while True:
                # n = 1 is Bernoulli distribution
                param = torch.tensor(np.random.binomial(n=1, p=0.5, size=(d)).astype(np.float32)) * self.config['r']
                if torch.linalg.norm(param, 2) > 0: break
            param = param/torch.linalg.norm(param, 2)  # l2 norm
            weight.data = param  # initialize weights as [0, 1]
            print(f'initial weights ({p_i}th cluster):', weight.data)

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
        m = self.config['m']  # number of total machines
        n = self.config['n']  # number of data points of each machine
        m_n = self.config['m_n']  # number of normal machines
        m_b = self.config['m_b']  # number of byzantine machines

        # flag_machines = np.asarray(
        #     [True if self.dataset['data'][m_i][-1][0].startswith('normal') else False for m_i in range(m_n + m_b)])

        # for each machine (client), find p centroids
        params = []
        for m_i in range(m_n + m_b):  # use all the machines' data
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
    loss.backward() # just compute the grads, no any updates.

    loss_value = loss.item()
    weight = model.weight()
    d_weight = weight.grad.clone()  # get the gradients

    # for debugging
    # for i in range(X.shape[1]):
    #     t = 2/X.shape[0] * sum([(_y-_y2) *(-1 * _x[i]) for _y, _y2, _x in zip(y, y_target, X)])
    #     print(f'grad_w_{i}:', t, ', (y, y^*, x):', [(_y, _y2, _x[i]) for _y, _y2, _x in zip(y, y_target, X)])
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
    if sum(mask) ==0:
        tmp = torch.zeros_like(grads[0])
    elif sum(mask) == 1:
        tmp = grads[mask][0]
    else: # sum(mask) > 1:
        tmp, indices = torch.median(grads[mask], dim=0)

    return tmp


def gradient_trimmed_mean(scores, grads, beta):
    scores = np.asarray(scores)
    grads = torch.stack(grads)
    mask = (scores == 1)

    if sum(mask) == 0:
        print('sum(mask) = 0')
        tmp = torch.zeros_like(grads[0])
    elif sum(mask) == 1:
        print('sum(mask) = 1, ', grads[mask])
        tmp = grads[mask][0]
    else:  # sum(mask) > 1:
        # remove beta percent data and then compute the trimmed mean
        d = grads.shape[1] # dimensions
        tmp = torch.zeros_like(grads[0])
        for i in range(d): # for each dimension
            ts = sorted([(v, v[i]) for v in grads[mask]], key=lambda vs: vs[1], reverse=False)
            m = int(np.floor(len(ts) * beta))  # remove the m smallest and m biggest grads from ts
            if 2 * m >= len(ts):
                raise ValueError(f'beta({beta}) is too large!')
            elif m == 0:
                # print(i, d, grads[mask][:, i], flush=True)
                _tmp = torch.mean(grads[mask][:, i], dim=0)
            else:
                _tmp = torch.mean(torch.stack([vs[0][i] for vs in ts[m:-m]]), dim=0)
            tmp[i] = _tmp

    return tmp


class SimpleLinear(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, 1, bias=False)  # weights from Uniform(-, +)

    def weight(self):
        return self.linear1.weight

    def forward(self, x):
        return self.linear1(x).view(-1)  # 1 dim


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
