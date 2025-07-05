#!/bin/bash
#SBATCH -p gpu                   # partition
#SBATCH --array=0-11%4           # 12 задач, max 4 в параллель
#SBATCH --nodes=1                # одна нода на под-задачу
#SBATCH --ntasks=1               # одна task  (= один процесс python)
#SBATCH --cpus-per-task=4        # 4 CPU-потока под DataLoader, т.п.
#SBATCH --gpus-per-task=1        # 1 GPU
#SBATCH --mem=80G                # RAM на узел
#SBATCH -t 1-00:00:00            # 1 day wall-time
#SBATCH -o logs/%A_%a.out        # stdout+stderr > logs/JobID_ArrayID.out
#SBATCH -e logs/%A_%a.err

# --- список датасетов в том же порядке, что в YAML ---
datasets=( delucionqa finqa cuad covidqa emanual expertqa 
           hagrid hotpotqa msmarco pubmedqa tatqa techqa )

ds=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "[`date`] Starting dataset  $ds  (task $SLURM_ARRAY_TASK_ID)"
srun python main.py 1 20 35 --dataset "$ds"
echo "[`date`] Finished dataset $ds"
