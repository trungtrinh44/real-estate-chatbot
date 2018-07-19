PYTHONPATH=. python ./data_utils/combine_all_texts.py
PYTHONPATH=. python ./data_utils/build_tokenizer.py
PYTHONPATH=. python ./data_utils/call_fasttext.py
PYTHONPATH=. python ./data_utils/process_train_data.py
fdupes -rdN ./output/data