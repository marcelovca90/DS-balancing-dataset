#!/bin/bash

# OpenML datasets
# - binary:
#   - 37        diabetes                        768x9x2
#   - 44        spambase                        4601x58x2
#   - 1462      banknote-authentication         1372x6x2
#   - 1479      hill-valley                     1212x101x2
#   - 1510      wdbc                            569x31x2
# - multiclass:
#   - 23        contraceptive-method-choice     1473x10x10
#   - 181       yeast                           1484x9x10
#   - 1466      cardiotocography                2126x24x10
#   - 40691     wine-quality                    1599x12x6
#   - 40975     car                             1728x7x4
# - multilabel:
#   - 41465     emotions                        593x78x6
#   - 41468     image                           2000x140x5
#   - 41470     reuters                         2000x250x7
#   - 41471     scene                           2407x300x6
#   - 41473     yeast                           2417x117x14

datasets=(37 44 1462 1479 1510 23 181 1466 40691 40975 41465 41468 41470 41471 41473)
seeds=(23 41 13 47 53 37 47 2 67 5 19 19 17 37 13)

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
python -m pip install --upgrade setuptools wheel
python -m pip install pandas scikit-learn scikit-multilearn optuna imbalanced-learn
python -m pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install autogluon.tabular[all] sdv

# Raise exception in case CTGANSynthesizer is likely to be slow
# https://github.com/sdv-dev/SDV/issues/1658
# https://github.com/sdv-dev/SDV/issues/1657
sed -i 's/, NotFittedError/, NotFittedError, SamplingError/' venv-autogluon/lib/python3.8/site-packages/sdv/single_table/ctgan.py
sed -i 's/print(  # noqa: T001/raise SamplingError(/' venv-autogluon/lib/python3.8/site-packages/sdv/single_table/ctgan.py

# Execution
for ((i=0; i<${#datasets[@]}; i++)); do 

    dataset_id="${datasets[i]}"
    target_name="${targets[i]}"
    seed="${seeds[i]}"

    echo ======== Execution ========
    echo Started processing dataset $id at $(date).

    python ./pipeline_optuna_autobalancing.py $dataset_id autogluon $seed

    echo Finished processing dataset $dataset_id at $(date).

    rm -rf ./AutogluonModels/

done

echo Script execution finished at $(date).
