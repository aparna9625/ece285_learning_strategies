MODEL SEARCH
python train_search_coop_pretrain.py

copy the best genotype from search into genotypes.py and pass it as argument (--arch) for training

MODEL TRAINING
python train.py

pass the directory to the saved model weights for testing (--model_path)

MODEL TESTING
python test.py
