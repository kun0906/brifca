from functools import partial

print = partial(print, flush=True)

import argparse
import json
import os
import pickle
import copy

import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import scipy
from util import *

# LR_DECAY = True
LR_DECAY = False

# Check if GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use the first available GPU
    print("n_GPUs:", torch.cuda.device_count())
    print("GPU: ", torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device("cpu")  # Use CPU if GPU is not available
print(f'CUDA is available: {DEVICE}')

parser = argparse.ArgumentParser()
parser.add_argument("--project_dir", type=str, default="alpha_01-beta_01")
parser.add_argument("--dataset_dir", type=str, default="alpha_01-beta_01")
parser.add_argument("--alg_method", type=str, default="proposed")
parser.add_argument("--update_method", type=str, default="mean")
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--data_seed", type=int, default=0)
parser.add_argument("--train_seed", type=int, default=0)
parser.add_argument("--config_override", type=str, default="")
args = parser.parse_args()

seed = args.data_seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


def main():
    config = get_config()

    # the way how we compute the weights on the server
    config['alg_method'] = args.alg_method
    config['update_method'] = args.update_method
    config['train_seed'] = config['data_seed'] = args.data_seed
    config['n_epochs'] = args.n_epochs
    config['project_dir'] = os.path.join(args.project_dir, config['alg_method'], config['update_method'],
                                         str(config['n_epochs']), str(config['data_seed']))
    config['beta'] = config['alpha']
    # reset m based
    # compute number of normal and Byzantine machines
    # 4 clusters, and each one has 60000 training images. Each client has 100 images.
    # 100 * 2400/4 = 60000
    config['m_n'] = 2400  # make sure that config['m_n'] % config['p] == 0
    # int(np.round(config['m'] * (1 - config['alpha'])))  # compute number of normal machines
    config['m'] = int(config['m_n'] / (1 - config['alpha']))
    m_b = config['m'] - config['m_n']  # compute number of Byzantine machines
    config['m_b'] = (m_b) // config['p'] * config['p']  # make sure that config['m_b'] % config['p] == 0
    config['m'] = config['m_n'] + config['m_b']  # update  config['m'] based on new m_n and m_b

    print("final config:", config)

    exp = TrainMNISTCluster(config)
    exp.setup()
    exp.run()


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


class TrainMNISTCluster(object):
    def __init__(self, config):
        self.seed = config['train_seed']
        self.random_state = self.seed
        self.config = config

        assert self.config['m_n'] % self.config['p'] == 0

    def setup(self):

        os.makedirs(self.config['project_dir'], exist_ok=True)
        # update_method = self.config['update_method']
        self.result_fname = os.path.join(self.config['project_dir'], f'results.pickle')
        self.checkpoint_fname = os.path.join(self.config['project_dir'], f'checkpoint.pt')

        self.setup_datasets()
        self.setup_models()

        self.epoch = None
        self.lr = None

    def setup_datasets(self):

        np.random.seed(self.config['data_seed'])

        # generate indices for each dataset
        # also write cluster info

        MNIST_TRAINSET_DATA_SIZE = 60000
        MNIST_TESTSET_DATA_SIZE = 10000

        # np.random.seed(self.config['data_seed'])

        cfg = self.config

        self.dataset = {}

        dataset = {}
        dataset['data_indices'], dataset['cluster_assign'] = \
            self._setup_dataset(MNIST_TRAINSET_DATA_SIZE, cfg['p'], cfg['m_n'], cfg['m_b'], cfg['n'], random=True)
        (X, y) = self._load_MNIST(train=True)
        dataset['X'] = X
        dataset['y'] = y
        self.dataset['train'] = dataset

        dataset = {}
        # no Byzantine machines in the test phrase
        dataset['data_indices'], dataset['cluster_assign'] = \
            self._setup_dataset(MNIST_TESTSET_DATA_SIZE, cfg['p'], cfg['m_test'], 0, cfg['n'], random=True)
        (X, y) = self._load_MNIST(train=False)
        dataset['X'] = X
        dataset['y'] = y
        self.dataset['test'] = dataset

        # import ipdb; ipdb.set_trace()

    def _setup_dataset(self, num_data, p, m_n, m_b, n, random=True):

        assert (m_n // p) * n == num_data

        dataset = {}

        cfg = self.config

        data_indices = []
        cluster_assign = []

        m_per_cluster = m_n // p
        for p_i in range(p):

            if random:
                ll = list(np.random.permutation(num_data))
            else:
                ll = list(range(num_data))

            ll2 = chunkify(ll, m_per_cluster)  # splits ll into m lists with size n
            data_indices += ll2

            cluster_assign += [p_i for _ in range(m_per_cluster)]

        data_indices = np.array(data_indices)
        cluster_assign = np.array(cluster_assign)
        assert data_indices.shape[0] == cluster_assign.shape[0]
        assert data_indices.shape[0] == m_n

        # Only use only distribution to mimic the data generated by Byzantine machines:
        if m_b > 0:
            # p outlier clusters, where each one has m_b//p Byzantine machines, and each machine has n outlier images.
            m_per_b_cluster = m_b // p
            b_data_indices = []
            b_cluster_assign = []

            for p_i in range(p):
                if random:
                    ll = list(np.random.permutation(num_data))
                else:
                    ll = list(range(num_data))
                ll = ll[:m_per_b_cluster * n]  # only have m_b*n outlier images
                ll2 = chunkify(ll, m_per_b_cluster)  # splits ll into m lists with size n
                b_data_indices += ll2

                b_cluster_assign += [p_i + p for _ in range(m_per_b_cluster)]

            data_indices = np.concatenate([data_indices, np.array(b_data_indices)], axis=0)
            cluster_assign = np.concatenate([cluster_assign, np.array(b_cluster_assign)], axis=0)
            assert data_indices.shape[0] == cluster_assign.shape[0]
            assert data_indices.shape[0] == m_b + m_n

        return data_indices, cluster_assign

    def _load_MNIST(self, train=True):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #   (0.1307,), (0.3081,))
        ])
        if train:
            mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
        else:
            mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

        dl = DataLoader(mnist_dataset)

        X = dl.dataset.data  # (60000,28, 28)
        y = dl.dataset.targets  # (60000)

        # normalize to have 0 ~ 1 range in each pixel

        X = X / 255.0

        return X.to(DEVICE), y.to(DEVICE)

    def setup_models(self):
        np.random.seed(self.config['train_seed'])
        torch.manual_seed(self.config['train_seed'])

        p = self.config['p']

        self.models = [SimpleLinear(h1=self.config['h1']).to(DEVICE) for p_i in
                       range(p)]  # p models with p different params of dimension(1,d)

        self.criterion = torch.nn.CrossEntropyLoss()

        # import ipdb; ipdb.set_trace()

    def run(self):
        n_epochs = self.config['n_epochs']
        lr = self.config['lr']

        results = []

        # epoch -1
        self.epoch = -1

        result = {}
        result['epoch'] = -1

        # cluster initialization
        t0 = time.time()
        res = self.test(train=True)  # change result['train']['cluster_assign'] to each machine
        t1 = time.time()
        res['infer_time'] = t1 - t0
        result['train'] = res

        self.print_epoch_stats(res)

        t0 = time.time()
        res = self.test(train=False)
        t1 = time.time()
        res['infer_time'] = t1 - t0
        result['test'] = res
        self.print_epoch_stats(res)
        results.append(result)

        # this will be used in next epoch
        cluster_assign = result['train']['cluster_assign']

        for epoch in range(n_epochs):

            self.epoch = epoch

            result = {}
            result['epoch'] = epoch

            lr = self.lr_schedule(epoch)
            result['lr'] = lr

            t0 = time.time()
            result['train'] = self.train(cluster_assign, lr=lr)  # update local and global weights
            t1 = time.time()
            train_time = t1 - t0

            t0 = time.time()
            res = self.test(train=True)  # update result['train']['cluster_assign']
            t1 = time.time()
            res['infer_time'] = t1 - t0
            res['train_time'] = train_time
            res['lr'] = lr
            result['train'] = res  # update result['train']

            self.print_epoch_stats(res)

            t0 = time.time()
            res = self.test(train=False)
            t1 = time.time()
            res['infer_time'] = t1 - t0
            result['test'] = res  # update result['test']
            self.print_epoch_stats(res)

            results.append(result)

            # this will be used in next epoch's gradient update
            cluster_assign = result['train']['cluster_assign']  # it will be changed in each epoch

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                with open(self.result_fname, 'wb') as outfile:
                    pickle.dump(results, outfile)
                    print(f'result written at {self.result_fname}')
                self.save_checkpoint()
                print(f'checkpoint written at {self.checkpoint_fname}')

        # import ipdb; ipdb.set_trace()

        with open(self.result_fname, 'wb') as outfile:
            pickle.dump(results, outfile)
            print(f'result written at {self.result_fname}')

        # plot the results
        self.plot_res(self.result_fname)

    def plot_res(self, results_fname):

        with open(results_fname, 'rb') as f:
            results = pickle.load(f)

        update_method = self.config['update_method']
        n_epochs = len(results)
        for plot_metric in ['acc', 'loss', 'infer_time', 'cl_acc']:
            train = []
            test = []
            for i in range(n_epochs):
                # if i == 0: continue
                vs = results[i]
                train.append(vs['train'][plot_metric])
                test.append(vs['test'][plot_metric])

            # plot the results
            is_show = True
            if is_show:
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
                # markers = ['*', '^', 'o', '+']
                # colors = ['g', 'b', 'purple', 'cyan']
                plt.plot(range(n_epochs), train, 'g*-', label=f"train-{plot_metric}")
                plt.plot(range(n_epochs), test, 'b^-', label=f"test-{plot_metric}")

                x_label = 'Epoch'
                plt.xlabel(f"{x_label}")
                plt.ylabel(f"{y_label}")
                plt.title(update_method)
                plt.legend(loc="upper left")

                # plt.xscale("log")
                # # plt.yscale("log")
                plt.tight_layout()

                # f = f"{out_dir}/n_{ns[0]}-{x_label}_{plot_metric}"
                f = f'{results_fname}-{plot_metric}.png'
                plt.savefig(f"{f}.png", dpi=100)
                # plt.savefig(f"{f}.eps", format="eps", dpi=100)
                # plt.savefig(f"{f}.svg", format="svg", transparent=True)
                print(f)
                plt.show()
                plt.clf()
                plt.close()

        # # import ipdb; ipdb.set_trace()

    def lr_schedule(self, epoch):
        if self.lr is None:
            self.lr = self.config['lr']

        if epoch % 50 == 0 and epoch != 0 and LR_DECAY:
            self.lr = self.lr * 0.1

        return self.lr

    def print_epoch_stats(self, res):
        if res['is_train']:
            data_str = 'tr'
        else:
            data_str = 'tst'

        if 'train_time' in res:
            time_str = f"{res['train_time']:.3f}sec(train) {res['infer_time']:.3f}sec(infer)"
        else:
            time_str = f"{res['infer_time']:.3f}sec"

        if 'lr' in res:
            lr_str = f" lr {res['lr']:4f}"
        else:
            lr_str = ""

        str0 = f"Epoch {self.epoch} {data_str}: loss {res['loss']:.3f} acc {res['acc']:.3f} clct{res['cl_ct']}{lr_str} {time_str}"

        print(str0)

    def train(self, cluster_assign, lr):
        VERBOSE = 0

        cfg = self.config
        m = self.config['m']  # number of total machines
        m_n = self.config['m_n']  # number of normal machines
        m_b = self.config['m_b']  # number of byzantine machines
        p = cfg['p']
        tau = cfg['tau']

        # run local update
        t0 = time.time()

        # flag_machines = np.asarray(
        #     [True if self.dataset['cluster_assign'][m_i] < p else False for m_i in range(m_n + m_b)])

        updated_models = []
        for m_i in range(m_n + m_b):  # each machine assignment won't update.
            if VERBOSE and m_i % 100 == 0: print(f'm {m_i}/{m} processing \r', end='')

            (X, y) = self.load_data(m_i)  # rotate images for normal machine and switch labels for Byzantine machine

            p_i = cluster_assign[m_i]
            # if p_i >= p: it won't happen, because we already assign each machine to each cluster
            model = copy.deepcopy(self.models[p_i])
            # if flag_machines[m_i]: # normal machines
            #     pass
            # else:   # abnormal machines
            #     pass
            for step_i in range(tau):  # train the model multiple times

                y_logit = model(X).to(DEVICE)
                loss = self.criterion(y_logit, y).to(DEVICE)

                model.zero_grad()
                loss.backward()
                self.local_param_update(model, lr)

            model.zero_grad()

            updated_models.append(model)

        t02 = time.time()
        # print(f'running single ..took {t02-t01:.3f}sec')

        t1 = time.time()
        if VERBOSE: print(f'local update {t1 - t0:.3f}sec')

        # apply gradient update
        t0 = time.time()

        local_models = [[] for p_i in range(p)]
        for m_i in range(m_n + m_b):
            # if flag_machines[m_i]:  # normal machines
            #     pass
            # else:  # abnormal machines
            #     pass
            p_i = cluster_assign[m_i]
            local_models[p_i].append(updated_models[m_i])

        for p_i, models in enumerate(local_models):
            if len(models) > 0:
                self.global_param_update(models, self.models[p_i])
        t1 = time.time()

        if VERBOSE: print(f'global update {t1 - t0:.3f}sec')

    def check_local_model_loss(self, local_models):
        # for debugging
        m = self.config['m']

        losses = []
        for m_i in range(m):
            (X, y) = self.load_data(m_i)
            y_logit = local_models[m_i](X).to(DEVICE)
            loss = self.criterion(y_logit, y).to(DEVICE)

            losses.append(loss.item())

        return np.array(losses)

    def get_inference_stats(self, train=True):
        cfg = self.config
        if train:
            m = cfg['m']
            assert m == cfg['m_n'] + cfg['m_b']
            dataset = self.dataset['train']
        else:
            m = cfg['m_test']
            dataset = self.dataset['test']

        p = cfg['p']

        num_data = 0
        losses = {}
        corrects = {}
        for m_i in range(m):
            (X, y) = self.load_data(m_i, train=train)  # load batch data rotated
            for p_i in range(p):
                y_logit = self.models[p_i](X.to(DEVICE)).to(DEVICE)
                loss = self.criterion(y_logit, y).to(DEVICE)  # loss of
                n_correct = self.n_correct(y_logit, y)

                losses[(m_i, p_i)] = loss.item()
                corrects[(m_i, p_i)] = n_correct

            num_data += X.shape[0]

        # calculate loss and cluster the machines
        cluster_assign = []
        for m_i in range(m):
            machine_losses = [losses[(m_i, p_i)] for p_i in range(p)]
            min_p_i = np.argmin(machine_losses)
            cluster_assign.append(min_p_i)

        # calculate optimal model's loss, acc over all models
        min_corrects = []
        min_losses = []
        for m_i, p_i in enumerate(cluster_assign):
            min_loss = losses[(m_i, p_i)]
            min_losses.append(min_loss)

            min_correct = corrects[(m_i, p_i)]
            min_corrects.append(min_correct)

        loss = np.mean(min_losses)
        acc = np.sum(min_corrects) / num_data

        # check cluster assignment acc
        cl_acc = np.mean(np.array(cluster_assign) == np.array(dataset['cluster_assign']))
        cl_ct = [np.sum(np.array(cluster_assign) == p_i) for p_i in range(p)]

        res = {}  # results
        # res['losses'] = losses
        # res['corrects'] = corrects
        res['cluster_assign'] = cluster_assign
        res['num_data'] = num_data
        res['loss'] = loss
        res['acc'] = acc
        res['cl_acc'] = cl_acc
        res['cl_ct'] = cl_ct
        res['is_train'] = train

        # import ipdb; ipdb.set_trace()

        return res

    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit.data, 1)
        correct = (predicted == y).sum().item()

        return correct

    def load_data(self, m_i, train=True):
        # this part is very fast since its just rearranging models
        cfg = self.config

        if train:
            dataset = self.dataset['train']
        else:
            dataset = self.dataset['test']

        indices = dataset['data_indices'][m_i]
        p_i = dataset['cluster_assign'][m_i]

        X_batch = dataset['X'][indices]
        y_batch = dataset['y'][indices]
        if p_i >= cfg['p']:  # for Byzantine machines:change here? 1) 0, 90, 180, 270. 2) 45, 135, 225, 315.
            # switch the label
            switch_labels = {9: 0, 8: 1, 7: 2, 6: 3, 5: 4, 4: 5, 3: 6, 2: 7, 1: 8, 0: 9}
            y_batch3 = torch.tensor([switch_labels[v] for v in torch.Tensor.cpu(y_batch).numpy()]).long()

            if cfg['p'] == 4:
                k = int(p_i)-cfg['p']
            # elif cfg['p'] == 2:
            #     k = (p_i % 2) * 2
            # elif cfg['p'] == 1:
            #     k = 0
            else:
                raise NotImplementedError("only p=1,2,4 supported")

            # X_batch2 = torchvision.transforms.functional.rotate(X_batch, (k*90)+45).to(DEVICE)
            # # X_batch2 = torch.rot90(X_batch, k=int(k), dims=(1, 2)).to(DEVICE)
            # X_batch3 = X_batch2.reshape(-1, 28 * 28).to(DEVICE)
            # # X_batch3 = 5 + X_batch3  # reverse the noise
            # # X_batch3 = X_batch3 + torch.FloatTensor(X_batch3.shape).uniform_(0, 1).to(DEVICE) # add [0, 1] values
            # # import matplotlib.pyplot as plt
            # # plt.imshow(X_batch2[0])
            # # plt.show()
            # # import ipdb; ipdb.set_trace()
            n_images = X_batch.shape[0]//cfg['p']
            X_batch2 = X_batch.clone().detach()
            for i in range(cfg['p']):
                _indices=range(i*n_images, (i+1)*n_images, 1)
                X_batch2[_indices,:] = torchvision.transforms.functional.rotate(X_batch[_indices,:], (i * 90) + 45).to(DEVICE)

            X_batch3 = X_batch2.reshape(-1, 28 * 28).to(DEVICE)
            return X_batch3.to(DEVICE), y_batch3.to(DEVICE)

        else:  # for normal machine

            # k : how many times rotate 90 degree
            # k =1 : 90 , k=2 180, k=3 270

            if cfg['p'] == 4:
                k = p_i
            elif cfg['p'] == 2:
                k = (p_i % 2) * 2
            elif cfg['p'] == 1:
                k = 0
            else:
                raise NotImplementedError("only p=1,2,4 supported")

            X_batch2 = torch.rot90(X_batch, k=int(k), dims=(1, 2)).to(DEVICE)
            X_batch3 = X_batch2.reshape(-1, 28 * 28).to(DEVICE)  #

            # import ipdb; ipdb.set_trace()

            return X_batch3, y_batch

    def local_param_update(self, model, lr):

        # gradient update manually

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad

        model.zero_grad()

        # import ipdb; ipdb.set_trace() # we need to check the output of name, check if duplicate exists

    def global_param_update(self, local_models, global_model):

        update_method = self.config['update_method']
        print('global_param_update: ', update_method)
        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = [torch.zeros_like(param.data)]
                # weights[name] += param.data
                weights[name].append(param.data)

        for name, param in global_model.named_parameters():
            ws = torch.stack(weights[name], dim=0)
            if update_method == 'mean':  # coordinate_wise_mean
                # average the grads
                # weights[name] /= len(local_models)
                ws = torch.mean(ws, dim=0)
            elif update_method == 'median':  # coordinate_wise_median
                ws, idx = torch.median(ws, dim=0)
            elif update_method == 'trimmed_mean':  # trimmed_mean
                beta = self.config['beta']  # parameter for trimmed mean
                # ws = trimmed_mean_weights(ws, beta)
                arr = scipy.stats.trim_mean(torch.Tensor.cpu(ws).numpy(), proportiontocut=beta, axis=0)
                ws = torch.from_numpy(arr).to(DEVICE)
            else:
                raise NotImplementedError(f'{update_method}')

            # param.data = weights[name]
            param.data = ws.to(DEVICE)
        # # import ipdb; ipdb.set_trace()

    def test(self, train=False):

        return self.get_inference_stats(train=train)

    def save_checkpoint(self):
        models_to_save = [model.state_dict() for model in self.models]
        torch.save({'models': models_to_save}, self.checkpoint_fname)


class SimpleLinear(torch.nn.Module):

    def __init__(self, h1=2048):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    # def weight(self):
    #     return self.linear1.weight


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
