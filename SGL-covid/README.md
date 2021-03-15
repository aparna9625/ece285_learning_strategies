## Code for Small Group Learning on COVID-CT dataset

1. Download COVID-CT dataset from https://github.com/UCSD-AI4H/COVID-CT and unzip the files into the covid_data folder
    ```
    cd covid_data/
    unzip CT_COVID.zip && unzip CT_NonCOVID.zip
    ```
2. Perform model search
    ```
    python train_search_coop_pretrain.py
    ```

3. Copy the best genotype from search logs into genotypes.py using a unique identifier 

3. Perform model training for image classification on searched model
    ```
    python train.py --arch <genotype_name>
    ```

4. Perform model testing
    ```
    python test.py --model_path <path to trained model weights>
    ```
