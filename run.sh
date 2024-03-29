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


password=YOUR_SUDO_PASSWORD
datasets=("openml_37" "openml_44" "openml_1462" "openml_1479" "openml_1510" "openml_23" "openml_181" "openml_1466" "openml_40691" "openml_40975" "openml_41465" "openml_41468" "openml_41470" "openml_41471" "openml_41473")
targets=("class" "class" "Class" "Class" "Class" "Contraceptive_method_used" "class_protein_localization" "Class" "class" "class" "class_ps" "class_ps" "class_ps" "class_ps" "class_ps")
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

for ((i=0; i<${#datasets[@]}; i++)); do 

    dataset_name="${datasets[i]}"
    target_name="${targets[i]}"
    seed="${seeds[i]}"

    echo ======== Processing ========
    echo Started processing dataset $id at $(date).

    # AutoGluon
    echo ======== AutoGluon ========
    python -m venv venv-autogluon
    source ./venv-autogluon/bin/activate
    python -m pip install --upgrade pip
    python -m pip install --upgrade setuptools wheel future autogluon
    python -m pip install optuna imbalanced-learn sdmetrics sdv torch
    python ./pipeline_optuna_autobalancing.py $dataset_name $target_name autogluon $seed

    # Delete AutoGluon models from last execution
    rm -rf ./AutogluonModels/

    echo Finished processing dataset $dataset_name at $(date).

done

echo Script execution finished at $(date).
