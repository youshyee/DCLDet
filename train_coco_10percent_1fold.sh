set -x
FileName=${0##*/}
NAME=${FileName%.*}
EXP_DIR=workdir/${NAME}
FOLD=1
PERCENT=10
python -m torch.distributed.run --nproc_per_node=4 main.py $NAME $PERCENT $FOLD --batch_size 1 --output_dir ${EXP_DIR} --eval \
--data_res 800 \
--lr 0.0002 \
--unsup_weight_end 4 \
--augplus
