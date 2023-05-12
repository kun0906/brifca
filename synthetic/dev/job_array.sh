#!/bin/bash
#SBATCH --job-name=cxx_mpi_omp   # create a short name for your job
#SBATCH --output=output.txt
#SBATCH --error=err.txt
##SBATCH --array=1-2                    # how many tasks in the array
#SBATCH --array=0,100,200           # 3 tasks with 3 different task IDs: SLURM_ARRAY_TASK_ID
#SBATCH --ntasks-per-node=3      # total number of tasks per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=ky8517@princeton.edu

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
cd /scratch/gpfs/ky8517/ifca/synthetic/dev
module load anaconda3/2021.11

python3 -V

#srun python3 task1.py > task1.txt 2>&1 &
#srun python3 task2.py > task2.txt 2>&1 &

# only run on one node
#python3 task1.py > task1.txt 2>&1 &
#python3 task2.py > task2.txt 2>&1 &

# Run python script with a command line argument
python task1.py --array_task_id $SLURM_ARRAY_TASK_ID > task1.txt 2>&1 &

wait
echo 'done'