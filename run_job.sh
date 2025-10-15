#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./run_name.out.%j                                    # File name for standard output
#SBATCH -e ./run_name.err.%j                                    # File name for error output
#SBATCH -J run_name
# Queue:
#SBATCH --nodes=1                                               # needs to match Trainer(num_nodes=)
#SBATCH --constraint="gpu"                                               
#SBATCH --gres=gpu:a100:4                                       # Specify number of GPUs needed for the job
#SBATCH --ntasks-per-node=4                                     # this needs to match Trainer(devices=)
#SBATCH --mem=125000                                    
#SBATCH --cpus-per-task=18                                      # needs to match num_workers
#SBATCH --account=$USER                                         # Can repalce with own username but this should work
#SBATCH --mail-type=all
#SBATCH --mail-user=email                                       # Add oyur own email address for notificaitons on job
#
# wall clock limit
#SBATCH --time=04:00:00                                         # Set limit on the total run time

set -e                                                          # Makes script exit immediately if any command returns non zero status (fails)

# Load necessary modules (if required)
module purge
module load apptainer  cuda/12.6 

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run the training script
srun apptainer exec --nv \
        --bind /u/$USER/code:/app/scripts \
        --bind /ptmp/$USER/data:/app/data \
        --bind /u/$USER/tb_logs:/app/tb_logs \
        --bind /u/$USER/outputs_3:/app/outputs \
        nvidia_pytorch_2.sif \
        python3 /app/scripts/run_job.py --batch-size 8 --max-epochs 150 --model Unet --data-dir "/app/data/bing_300m_semantic_segmentation_april_split"

srun apptainer exec --nv \
        --bind /u/$USER/code:/app/scripts \
        --bind /ptmp/$USER/data:/app/data \
        --bind /u/$USER/tb_logs:/app/tb_logs \
        --bind /u/$USER/outputs_3:/app/outputs \
        nvidia_pytorch_2.sif \
        python3 /app/scripts/run_job.py --batch-size 8 --max-epochs 150 --model Segformer --data-dir "/app/data/bing_300m_semantic_segmentation_april_split"

srun apptainer exec --nv \
        --bind /u/$USER/code:/app/scripts \
        --bind /ptmp/$USER/data:/app/data \
        --bind /u/$USER/tb_logs:/app/tb_logs \
        --bind /u/$USER/outputs_3:/app/outputs \
        nvidia_pytorch_2.sif \
        python3 /app/scripts/run_job.py --batch-size 8 --max-epochs 150 --model MAnet --data-dir "/app/data/bing_300m_semantic_segmentation_april_split"

