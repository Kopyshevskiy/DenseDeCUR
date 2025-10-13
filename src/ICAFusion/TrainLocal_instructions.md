### Training locale con GPU NVIDIA

Crea env 
```bash
conda create -n icafusion-env python=3.8 -y
conda activate icafusion-env
```

installa versioni pytorch compatibili con versione cuda scelta
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


Go to ICAFusion dir
```bash
cd ICAFusion
```

Installa i requirements di progetto
```bash
pip install -r requirements.txt
```

Aggiungi questi pacchetti
```bash
pip install requests, einops, timm, wandb
```


Prepara un yaml in questa forma:
```text
path: /FULL PATH ALLA CARTELLA CHE CONTIENE IL DATASET DIVISO IN VISIBLE INFRARED E LABELS 

train_rgb: /Absolute_path.. /visible/train
val_rgb: /Absolute_path.. /visible/test
train_ir: /Absolute_path.. /infrared/train
val_ir: /Absolute_path.. /infrared/test
train_labels: /Absolute_path.. /labels/train
val_labels: /Absolute_path.. /labels/test

nc: 1          numero di classi 
names: ['person']    label associata
```

Questo sara' il yaml passato come argomento in --data

Componi ora la chiamata in questo modo
```bash
python train.py \
  --weights "" \
  --cfg ./models/transformer/yolov5_ResNet50_Transfusion_kaist.yaml \
  --data ./data/kaist_icafusion_extrasmall.yaml \
  --hyp ./data/hyp.kaist.scratch.yaml \
  --epochs 1 \
  --batch-size 2 \
  --img-size 640 \
  --workers 1 \
  --project runs/train \
  --name icafusion_debug_small \
  --exist-ok
```

ricorda che in data hai i file yaml di config 


Con pretraining 

Inserire in --weights uno dei file presenti in ~/ICAFusion/final_checkpoints/...

```bash 
python train.py \
  --weights "/home/honey/ADL-Project/ICAFusion/final_checkpoints/icafusion_from_densecur.pth" \
  --cfg ./models/transformer/yolov5_ResNet50_Transfusion_kaist.yaml \
  --data ./data/kaist_icafusion_extrasmall.yaml \
  --hyp ./data/hyp.kaist.scratch.yaml \
  --epochs 1 \
  --batch-size 4 \
  --img-size 640 \
  --workers 1 \
  --project runs/train \
  --name icafusion_debug_small \
  --exist-ok
```