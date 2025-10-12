#!/bin/bash
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
# #SBATCH --constraint="[a100_80gb|h100_80gb]"
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00:00
#SBATCH --array=0-2
#SBATCH --nice=1000
#SBATCH --mail-user=ben.riegler@helmholtz-munich.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=high_d
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
source /home/haicu/ben.riegler/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu
cd /home/haicu/ben.riegler/work/IC-SBI/



mkdir -p SLURM_GPU/$experiment_name
output_log="SLURM_GPU/${experiment_name}/${SLURM_ARRAY_TASK_ID}_output.txt"
error_log="SLURM_GPU/${experiment_name}/${SLURM_ARRAY_TASK_ID}_error.txt"
bash_log="SLURM_GPU/${experiment_name}/${SLURM_ARRAY_TASK_ID}_bash.txt"
# Clear existing logs
rm -f "$output_log" "$error_log" "$bash_log"
echo "The ID of this task is ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"> "$bash_log"
echo "Calling script for ${experiment_name} - ${SLURM_ARRAY_TASK_ID}" >> "$bash_log"
echo "Script called at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$bash_log"
python -u main.py \
    --folder $folder \
    --experiment_name "$experiment_name" \
    --NUM_WORK $workers  \
    --T $T \
    --N_RUNS $N_RUNS \
    --dim_list $DIMS \
    --objective "$experiment_name" \
    --sim_or_real "True" \
    --methods "${method_args[@]}" \
    > "$output_log" 2> "$error_log"
echo "Called script for ${experiment_name}" >> "$bash_log"
echo "Script completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$bash_log"