set -x
OFFSET=$RANDOM
for percent in 1 5 10; do
    for fold in 1 2 3 4 5; do
        $(dirname "$0")/get_split.py --percent ${percent} --seed ${fold} --data-dir $1 --seed-offset ${OFFSET}
    done
done
