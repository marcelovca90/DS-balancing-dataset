#!/bin/bash

# +--------------------------------------------------------------------------------------------------------------+
# |                                 OpenML Datasets (https://www.openml.org/search?type=data)                    |
# +------------+--------+-----------------------------+----------------+-----------+----------+------------------+
# | Type       | ID     | Dataset Name                | Data Type      | Instances | Features | Classes (Labels) |
# |------------|--------|-----------------------------|----------------|-----------|----------|------------------|
# |            | 31     | credit-g                    | Mixed          | 1,000     | 20       | 2                |
# |            | 37     | diabetes                    | Quantitative   | 768       | 8        | 2                |
# |            | 44     | spambase                    | Quantitative   | 4,601     | 57       | 2                |
# | Binary     | 1462   | bank-note-authentication    | Quantitative   | 1,372     | 4        | 2                |
# |            | 1479   | hill-valley                 | Quantitative   | 1,212     | 100      | 2                |
# |            | 1510   | wdbc                        | Quantitative   | 569       | 30       | 2                |
# |            | 40945  | titanic                     | Mixed          | 1,309     | 13       | 2                |
# |------------|--------|-----------------------------|----------------|-----------|----------|------------------|
# |            | 23     | contraceptive-method-choice | Mixed          | 1,473     | 9        | 3                |
# |            | 36     | segment                     | Mixed          | 2,310     | 19       | 7                |
# |            | 54     | vehicle                     | Quantitative   | 846       | 18       | 4                |
# | Multiclass | 181    | yeast                       | Mixed          | 1,484     | 8        | 10               |
# |            | 1466   | cardiotocography            | Mixed          | 2,126     | 35       | 10               |
# |            | 40691  | wine-quality-red            | Quantitative   | 1,599     | 11       | 6                |
# |            | 40975  | car                         | Qualitative    | 1,728     | 6        | 4                |
# |------------|--------|-----------------------------|----------------|-----------|----------|------------------|
# |            | 285    | flags                       | Mixed          | 194       | 17       | 12 (103)         |
# |            | 41464  | birds                       | Mixed          | 645       | 260      | 19 (133)         |
# |            | 41465  | emotions                    | Mixed          | 593       | 72       | 6 (27)           |
# | Multilabel | 41468  | image                       | Quantitative   | 2,000     | 135      | 5 (20)           |
# |            | 41470  | reuters                     | Mixed          | 2,000     | 243      | 7 (25)           |
# |            | 41471  | scene                       | Quantitative   | 2,407     | 294      | 6 (15)           |
# |            | 41473  | yeast                       | Quantitative   | 2,417     | 103      | 14 (198)         |
# +------------+--------+-----------------------------+----------------+-----------+----------+------------------+

datasets=(37 44 1462 1479 1510 23 181 1466 40691 40975 41465 41468 41470 41471 41473)
seeds=(23 41 13 47 53 37 47 2 67 5 19 19 17 37 59)

echo Script execution started at $(date).

# Preparation
echo ======== Preparation ========
echo Started cleaning files from previous executions at $(date).
rm -rf __pycache* &> /dev/null
rm -rf artifacts/optuna_models/* &> /dev/null
rm -rf autobalancer_models* &> /dev/null
rm -rf autobalancer_results* &> /dev/null
rm -rf autobalancer_optuna_results* &> /dev/null
rm -rf Autogluon* &> /dev/null
rm -rf gama* &> /dev/null
rm -rf results* &> /dev/null
rm -rf structured* &> /dev/null
rm -rf venv-* &> /dev/null
rm artifacts/autobalancer_datasets/*_train.csv &> /dev/null
rm artifacts/autobalancer_datasets/*_test.csv &> /dev/null
rm artifacts/exec_logs/* &> /dev/null
rm artifacts/optuna_dbs/* &> /dev/null
rm artifacts/sdv_cache/* &> /dev/null
echo Finished cleaning files from previous executions at $(date).

# Virtual Environment
echo ======== Virtual Environment ========
python3.8 -m venv venv-autogluon
source ./venv-autogluon/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel pandas scikit-learn scikit-multilearn optuna imbalanced-learn "spacy<3.8" "blis<1.0" torch==1.12+cpu torchvision==0.13.0+cpu torchtext==0.13.0 "autogluon<1.2" sdv -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Raise exception in case CTGANSynthesizer is likely to be slow
# https://github.com/sdv-dev/SDV/issues/1658
# https://github.com/sdv-dev/SDV/issues/1657
sed -i 's/, NotFittedError/, NotFittedError, SamplingError/' venv-autogluon/lib/python3.8/site-packages/sdv/single_table/ctgan.py
sed -i 's/print(  # noqa: T001/raise SamplingError(/' venv-autogluon/lib/python3.8/site-packages/sdv/single_table/ctgan.py

# Execution
for ((i=0; i<${#datasets[@]}; i++)); do 

    dataset_id="${datasets[i]}"
    target_name="class"
    seed="${seeds[i]}"

    echo ======== Execution ========
    echo Started processing dataset $id at $(date).

    python ./py_pipeline_optuna_autobalancing.py $dataset_id autogluon $seed

    echo Finished processing dataset $dataset_id at $(date).

    rm -rf ./AutogluonModels/

done

echo Script execution finished at $(date).
