mkdir -p log_dir/runs/$1
cp output/*.pkl log_dir/runs/$1
python train.py --version=$1 --batch_size=$2 --num_epoch=$3 --word_vec=./data/wordvec_100.bin
