V0.0.8: Update geometric_median()
1. Add init_method=true_weights for geometric_median()
2. Add init_method=true_label for geometric_median()
3. Add init_method=60label for geometric_median()
4. Add alg_method=args.alg_method into main(n=args.n, d=args.d, update_method=args.update_method)
5. Update max_procs = 30

V0.0.7: Add init_method=true for geometric_median()
1. Add init_method=true for geometric_median()
2. Add verbose and don't print the weights 
3. run K=15 and m = 600


V0.0.6-2: Fix errors in geometric_kmeans()
1. Fix errors
    if inside_points.shape[0] == 0:
        # msg = f'too large radius:{radius}'
        # raise ValueError(msg)
        new_centroids[j] = np.mean(cluster_j_points, axis=0)
    else:
        new_centroids[j] = np.mean(inside_points, axis=0)
2. flatten the weights of SimplerLinear


V0.0.6-1: Update geometric_kmeans()
thres = np.quantile(_dists, q=0.95)
inside_points = cluster_j_points[_dists < thres]

V0.0.6: Add train_cluster_baseline and submit mutli-jobs onto HPC
1. Submit mutli-jobs onto HPC in main_sbatch.py 
2. Add train_cluster_baseline
   for each cluster, 
      1. Compute geometric_median
      2. Only keep the points inside the hyperball with a radius
      3. Compute the mean on the points kept. 
3. Plot mult-metric plots and update std_error: 2*\sigma/sqrt(n)
4. Move unused scripts to "dev"
5. Add "scp" into install.txt 



V0.0.5: Each Byzantine machine has its distribution
1. Each Byzantine machine has its distribution
2. Update y_label in plot
3. Reduce epoch=300 to 100
4. Add datetime to gen_data_and_train_cluster.py
5. Add install.txt and update requirement.txt 


V0.0.4:Update trimmed means
1. Update trimmed means in gradient_trimmed_mean()
    _tmp = torch.mean(grads[mask][:, i], dim=0)
2. Add more colors to plot_data()
 colors = ['r', 'g', 'b', 'black', 'm', 'brown', 'purple', 'yellow',
              'tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray',
              'tab:olive','tab:cyan',]
3. Update runner.summarize(force=args.force)
   ns = self.cfg['n']
   x_label = 'n' 
   xs = ns  # x-axis


V0.0.3: Change Binomial distribution to Bernoulli distribution
1. Change Binomial distribution to Bernoulli distribution
2. Normalize the estimated weights after each update
3. Modify calculate_loss_grad() for Byzantine machines 


V0.0.2: Add coordinate-wise median and trimmed mean 
1. Add coordinate-wise median and trimmed mean 
2. Initialize the weights/parameters with values that are closet to ground-truth
3. Add run_all.py and plots (Overwrite "summarize()") 
4. output everything to "output"


V0.0.1: Add Byzantine machines to synthetic data 
1. Add Byzantine machines to [gen_data_and_train_cluster.py](gen_data_and_train_cluster.py)
2. Add client_init() to train_cluster.py with 'random' and 'kmeans++'
3. Update Server grads with median instead of averaging means of grads 
