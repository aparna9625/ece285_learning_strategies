## Code for Small Group Learning on COVID-CT dataset

1. Download COVID-CT dataset from https://github.com/UCSD-AI4H/COVID-CT
--> unzip CT_COVID.zip and CT_NonCOVID.zip in the covid_data folder

2. To perform model search:
  python train_search_coop_pretrain.py

--> copy the best genotype from search into genotypes.py

3. To perform model training:
  python train.py --arch <genotype name>

4. To perform model testing:
  python test.py --model_path <path to trained model weights>
