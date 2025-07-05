#!/bin/bash
#SBATCH -p gpu                   # partition
#SBATCH --array=0-11%4           # 12 �����, max 4 � ���������
#SBATCH --nodes=1                # ���� ���� �� ���-������
#SBATCH --ntasks=1               # ���� task  (= ���� ������� python)
#SBATCH --cpus-per-task=4        # 4 CPU-������ ��� DataLoader, �.�.
#SBATCH --gpus-per-task=1        # 1 GPU
#SBATCH --mem=80G                # RAM �� ����
#SBATCH -t 1-00:00:00            # 1 day wall-time
#SBATCH -o logs/%A_%a.out        # stdout+stderr > logs/JobID_ArrayID.out
#SBATCH -e logs/%A_%a.err

# --- ������ ��������� � ��� �� �������, ��� � YAML ---
datasets=( delucionqa finqa cuad covidqa emanual expertqa 
           hagrid hotpotqa msmarco pubmedqa tatqa techqa )

ds=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "[`date`] Starting dataset  $ds  (task $SLURM_ARRAY_TASK_ID)"
srun python main.py 1 20 35 --dataset "$ds"
echo "[`date`] Finished dataset $ds"
