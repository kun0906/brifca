#!/bin/bash

out_dir="small_set-alpha_01-beta_01"
#out_dir="/u/ky8517/ifca/mnist/small_set-alpha_01-beta_01"
if [ ! -d "$out_dir" ]; then
  mkdir "$out_dir"
fi

#module load anaconda3/2021.11
#conda env list
#conda create --name py3104_ifca python=3.10.4
#conda activate py3104_ifca

pwd
python3 -V
uname -a
hostname -s

date
start=0
end=99
step=2
data_seeds=($(seq $start $step $end))
echo "${data_seeds[@]}"
alg_methods=('proposed' 'baseline')
update_methods=("mean" "median" "trimmed_mean")
for alg_method in "${alg_methods[@]}"; do
  for update_method in "${update_methods[@]}"; do
    for data_seed in "${data_seeds[@]}"; do
      cmd="python3 -u train_cluster_mnist_small_set.py --data-seed ${data_seed} --update_method ${update_method} --alg_method ${alg_method}"
      echo "$cmd"
      _out_dir="${out_dir}/${alg_method}/${update_method}/${data_seed}"
      if [ ! -d "$_out_dir" ]; then
        mkdir -p "$_out_dir"
      fi
      $cmd > "${_out_dir}/log.txt" 2>&1 &
    done
  done
done

date
wait
echo 'done'
