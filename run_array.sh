#!/bin/bash
#SBATCH -p gpu                  
#SBATCH --array=0-11%4          
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4        
#SBATCH --gpus-per-task=1        
#SBATCH --mem=80G                
#SBATCH -t 1-00:00:00            
#SBATCH -o logs/%A_%a.out        
#SBATCH -e logs/%A_%a.err


datasets=( delucionqa finqa cuad covidqa emanual expertqa 
           hagrid hotpotqa msmarco pubmedqa tatqa techqa )

ds=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "[`date`] Starting dataset  $ds  (task $SLURM_ARRAY_TASK_ID)"
srun python main.py 1 20 35 --dataset "$ds"
echo "[`date`] Finished dataset $ds"
