#!/bin/bash
#SBATCH --job-name=pretrain_DeCUR
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretrain_densedecur.out
#SBATCH --error=logs/pretrain_densedecur.err


echo "Starting"



module load Python/3.9.6-GCCcore-11.2.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1


source venv/bin/activate


export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUDA_LAUNCH_BLOCKING=1


python /mnt/proj3/eu-25-19/davide_secco/ADL-Project/DenseDeCUR/main.py \
  --dataset KAIST \
  --method DenseDeCUR \
  --densecl_stream thermal \
  --batch-size 128 \
  --epochs 200 \
  --print-freq 20 \
  --dim_common 96



  echo "Ending"
