export PYTHONPATH=.
python data_utils/process_train_data.py --output='output/test' --input='data/test' --word_tokenizer=trained_model/$1/word_tokenizer.pkl --char_tokenizer=trained_model/$1/char_tokenizer.pkl
fdupes -rdN ./output/test
python evaluate.py --model=trained_model/$1/ --version=$2 > eval/eval-$1_$2.txt
code eval/eval-$1_$2.txt