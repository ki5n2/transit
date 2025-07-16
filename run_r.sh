#!/bin/bash
#SBATCH --job-name=transit_
#SBATCH --output=/home1/rldnjs16/transit/logs/out/data_life_%j.out
#SBATCH --error=/home1/rldnjs16/transit/logs/err/data_life_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1          # 하나의 태스크가 사용할 CPU 코어 수를 48개로 설정

#SBATCH --partition=gpu4   # 예시
#SBATCH --nodelist=n097        # 사용할 노드 이름

##SBATCH --gres=gpu:a6000:1                    # GPU 1개 할당 (GPU가 필요한 경우)     
##SBATCH --gres=gpu:1
##SBATCH --gpus-per-task=rtx3090:1

##SBATCH --mem=8G
##SBATCH --time=01:00:00

module purge
module load R/4.3.1       # ✅ 핵심
which R

# R 스크립트 실행
Rscript /home1/rldnjs16/transit/ex.r
