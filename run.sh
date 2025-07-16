#!/bin/bash
#SBATCH --job-name=transit          # 작업 이름
#SBATCH --output=/home1/rldnjs16/transit/logs/out/data_life_%j.out
#SBATCH --error=/home1/rldnjs16/transit/logs/err/data_life_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1          # ntasks = nodes x ntasks-per-node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1          # 하나의 태스크가 사용할 CPU 코어 수를 48개로 설정

#SBATCH --partition=gpu6          # 사용할 파티션 이름 (클러스터 환경에 맞게 변경)
#SBATCH --nodelist=n036        # 사용할 노드 이름

##SBATCH --gres=gpu:rtx3090:1                     # GPU 1개 할당 (GPU가 필요한 경우)     
#SBATCH --gres=gpu:1
##SBATCH --gpus-per-task=rtx3090:1

##SBATCH --time=00:10:00                  # 최대 실행 시간 (형식: HH:MM:SS)
##SBATCH --mem=4G                         # 메모리 할당량

# 필요한 모듈 로드 (Python 및 CUDA 버전은 환경에 맞게 조정)
# module load python/3.11.2         # 클러스터 환경에서 필요한 프로그램을 불러오는 명령어(CUDA 11.8.0 버전을 사용, 이 명령어가 없으면 nvcc, nvidia-smi, torch.cuda 등이 작동 안 할 수 있음)
module load cuda/11.8.0

# 가상환경 사용 시 활성화 (필요한 경우)
# source ~/envs/myenv/bin/activate

# Conda 환경 활성화를 위해 conda 초기화 스크립트를 먼저 source합니다.
source /home1/rldnjs16/ENTER/etc/profile.d/conda.sh         # Conda 환경(ython 패키지(예: PyTorch, NumPy)와 환경(환경마다 다른 버전들)을 깔끔하게 관리해주는 툴)을 사용할 수 있게 초기화하는 명령어로, Conda를 사용하기 위한 준비 단계임

# 가상환경 활성화
conda activate city # city라는 conda 환경에서, 슈퍼컴퓨팅 쓸 준비 완료

# Python 스크립트 실행
python /home1/rldnjs16/transit/transit_sample3.py 
