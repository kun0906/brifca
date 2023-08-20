
import os
import pickle

file_name = '/Users/kun/Downloads/results.pkl'
# file_name = 'alpha_005-beta_005-20230515/K=2_our_algorithm.pkl'
file_name = 'alpha_005-beta_005-20230515/K=10_our_algorithm.pkl'
with open(file_name, 'rb') as f:
    data = pickle.load(f)

results = []
n = 100
for d in [20,50,100,200,500]:
    for update_method in ["mean","median","trimmed_mean"]:
        for alg_method in ["proposed"]:
            for data_seed in range(0, 100, 2):
                train_seed = 0
                prefix = f"p_15_m_600_alpha_0.050000_n_{n}_d_{d}_noise_scale_0.447200_r_1.000000_" \
                         f"alg_method_{alg_method}_update_method_{update_method}_beta_0.050000_" \
                         f"data_seed_{data_seed}_train_seed_{train_seed}_lr_0.010000"
                print(prefix)
                tmp_dir = 'OUT'
                tmp_file = os.path.join(tmp_dir, prefix, 'result.pkl')
                with open(tmp_file, 'rb') as f_:
                    result_  = pickle.load(f_)
                results.append(result_)
file_name = 'tmp.pkl'
with open(file_name, 'wb') as f:
    pickle.dump(results, f)

print(data)
