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
