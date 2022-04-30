set -x
FileName=${0##*/}
NAME=${FileName%.*}
EXP_DIR=workdir/${NAME}
FOLD=1
PERCENT=1
python -m torch.distributed.run --nproc_per_node=4 main.py $NAME $PERCENT $FOLD --batch_size 6 --output_dir ${EXP_DIR} --eval \
--data_res 800 \
--epochs 2080 \
--lr 0.0002 \
--warmup_epochs 2000 \
--lr_drop 2070 \
--unsup_weight_end 4 \
--augplus \
