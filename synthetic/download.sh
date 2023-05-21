#!/bin/bash

out_dir="/Users/kun/Projects/ifca/synthetic/alpha_01-beta_01-20230520"
if [ ! -d "$out_dir" ]; then
  mkdir "$out_dir"
fi

#K=2
#alg_method='proposed'
#rsync -avz --partial --progress ky8517@nobel.princeton.edu:"/u/ky8517/ifca/synthetic/output-K_${K}-${alg_method}-alpha_01-beta_01/results.pkl" "${out_dir}/K_${K}-${alg_method}.pkl"

ks=(2 5 10 15)
alg_methods=('proposed' 'baseline')
for alg_method in "${alg_methods[@]}"; do
  for K in "${ks[@]}"; do
    remote="ky8517@nobel.princeton.edu:\"/u/ky8517/ifca/synthetic/output-K_${K}-${alg_method}-alpha_01-beta_01/results.pkl\""
    local="${out_dir}/K_${K}-${alg_method}.pkl"
    cmd="rsync -avzh --partial --progress -e \"ssh\" $remote $local"
    echo "$cmd"
    $cmd
  done
done
