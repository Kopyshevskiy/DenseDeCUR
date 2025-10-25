#!/bin/bash
#SBATCH --job-name=setup_DenseDeCUR
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --error=logs/install.err
#SBATCH --output=logs/install.out

echo "[INFO] wassup!"

# manteniamo l'ambiente pulito
module purge
# carichiamo Python e PyTorch
module load Python/3.9.6-GCCcore-11.2.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 

cd ../DenseDeCUR

# step 1: crea (se serve) virtualenv
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtualenv with Python 3.9.6"
    python -m venv venv
fi

source venv/bin/activate
which python              
python --version

# step 2a: NON installo PyTorch via pip, perché uso quello del modulo
echo "[INFO] Skipping PyTorch pip install: using system module with CUDA 12.1"

# step 2b: installa torchvision e torchaudio compatibili con il modulo PyTorch
echo "[INFO] Installing torchvision / torchaudio compatible with PyTorch 2.1.2..."
pip install torchvision==0.16.2 torchaudio==2.1.2

# step 3: installa dipendenze del progetto
echo "[INFO] Installing project dependencies"
pip install --upgrade pip
pip install -r requirements.txt --verbose
pip install numpy<2.0
pip install opencv-contrib-python<4.7
pip install tensorboard
pip install diffdist
pip install typing-extensions
pip install opencv-python
pip install rasterio==1.3.9 
pip install albumentations
pip install opencv-torchvision-transforms-yuzhiyang
pip install matplotlib

# step 4: debug CUDA
echo "[INFO] Checking PyTorch GPU support..."
python -c "import torch; print('CUDA:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0)); print('Available:', torch.cuda.is_available()); print('Capability:', torch.cuda.get_device_capability())"

echo "[INFO] DeCUR Installation completed."




