
### DeCUR 

```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python main.py \
  --dataset KAIST \
  --method DeCUR \
  --densecl_stream thermal \
  --data-root ~/ADL-Project/kaist-cvpr15/images \
  --list-train ~/ADL-Project/Kaist_txt_lists/Training_onlyOn_Set05.txt \
  --batch-size 8 \
  --epochs 1 \
  --checkpoint-dir ./checkpoint/UnOfficial_DeCUR_from_lists \
  --print-freq 20
  ```

### DenseCL
```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python main.py \
  --dataset KAIST \
  --method DenseCL \
  --densecl_stream rgb \
  --data-root ~/ADL-Project/kaist-cvpr15/images \
  --list-train ~/ADL-Project/Kaist_txt_lists/Training_onlyOn_Set05.txt \
  --batch-size 8 \
  --epochs 1 \
  --checkpoint-dir ./checkpoint/UnOfficial_DenseCL_from_lists \
  --print-freq 20
  -- dim_common 96
  ```


### DenseDeCUR

```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python main.py \
  --dataset KAIST \
  --method DenseDeCUR \
  --densecl_stream rgb \
  --data-root ~/ADL-Project/kaist-cvpr15/images \
  --list-train ~/ADL-Project/Kaist_txt_lists/Training_onlyOn_Set05.txt \
  --batch-size 8 \
  --epochs 1 \
  --checkpoint-dir ./checkpoint/Official_DenseDeCUR_from_lists \
  --print-freq 20
  -- dim_common 96
  ```