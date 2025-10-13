# Creare env
```bash
source /mnt/proj3/eu-25-19/davide_secco/miniconda3/etc/profile.d/conda.sh
conda create -n icafusion
```

# Settare env
```bash
conda activate icafusion
cd ./ICAFusion
pip install -r requirements.txt
pip install requests, einops, timm
```