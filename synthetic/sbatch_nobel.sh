#!/bin/bash

# 1. sbatch_nobel.sh to execute in parallel
#   in each process, use python run_all.py to obtain each result
# 2. collect all the results by collect_all_results.py (if 'result.pkl' exists, then we gather it to the final results;
#   Otherwise, we will obtain the result from scratch)
# 3. then plot the results by plot_result.py





module purge
cd /u/ky8517/ifca/synthetic
module load anaconda3/2021.11
conda activate py3104_ifca
out_dir="OUT"
if [ ! -d "$out_dir" ]; then
    mkdir "$out_dir"
    echo "Folder created: $out_dir"
else
    echo "Folder already exists: $out_dir"
fi

pwd
python3 -V
uname -a

for n in 100; do
    for d in 20 50 100 200 500; do
        for update_method in "mean" "median" "trimmed_mean"; do
            for alg_method in "proposed"; do
                cnt=$((cnt + 1))
                name="n_${n}-d_${d}-${update_method}-${alg_method}"
                echo "Processing $name"
                PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 run_all.py --out_dir ${out_dir} -n $n -d $d --update_method ${update_method} --alg_method ${alg_method} > "${out_dir}/out_${name}.txt" 2>&1
            done
        done
    done
done

#PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 run_all.py -n $n -d $d --update_method ${update_method} --alg_method ${alg_method}

#wait


echo 'done'

