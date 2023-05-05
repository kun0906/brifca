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
