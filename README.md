# Dynamic-Balancing AutoML for Imbalanced Tabular Data with Adaptive Resampling and Complexity-Aware Analysis

Authors: *Marcelo V. C. Aragão, Tiago de M. Pereira, Mateus de F. Carvalho, Felipe A. P. de Figueiredo, and Samuel B. Mafra.*

## Abstract:
    Handling class imbalance is a fundamental challenge in supervised learning,
    particularly in real-world scenarios where minority classes are critical yet
    underrepresented. This paper presents a novel dynamic-balancing pipeline that
    enhances AutoML performance on imbalanced tabular datasets. The proposed
    approach integrates both traditional and generative resampling techniques with
    adaptive, class-specific thresholds, enabling automated and dataset-sensitive
    balancing strategies. To assess its generalizability, the pipeline is applied
    uniformly across binary, multiclass, and multilabel classification tasks. Each
    configuration is evaluated within an AutoML framework using performance and
    efficiency metrics, with outcomes validated through statistical testing and
    effect size analysis. The study also incorporates dataset complexity measures
    - including feature-label dependency and class overlap - to investigate how
    structural characteristics affect balancing efficacy. By combining principled
    resampling, exhaustive grid search, and rigorous evaluation, the pipeline
    enables more robust and efficient AutoML workflows. This work contributes a
    flexible and reproducible framework for addressing class imbalance, particu-
    larly in multilabel contexts, and establishes a foundation for scalable,
    complexity-aware resampling strategies in automated model development.

## Setup and Execution:

#### Python 3.8 installation
- `sudo add-apt-repository ppa:deadsnakes/ppa -y`
- `sudo apt install python3.8 python3.8-distutils python3.8-venv -y`
- `wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py`
- `python3.8 /tmp/get-pip.py`

#### Environment setup
- `cd ~/ && mkdir git && cd ~/git/`
- `git clone https://github.com/marcelovca90/DS-balancing-dataset.git`
- `cd DS-balancing-dataset`
- `pip install -r requirements.txt`

#### Experiments execution
- `chmod +x run.sh`
- `./run.sh | tee run.log`
- `jupyter lab`
- On Jupyter Lab:
  - Open and run `nb_complexity_metrics.ipynb`
  - Open and run `nb_performance_results.ipynb.ipynb`
  - Open and run `nb_statistical_tests.ipynb.ipynb`