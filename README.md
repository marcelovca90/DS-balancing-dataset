## Execute the following commands at the terminal

#### Python 3.8 installation
- `sudo add-apt-repository ppa:deadsnakes/ppa -y`
- `sudo apt install python3.8 python3.8-distutils python3.8-venv -y`
- `wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py`
- `python3.8 /tmp/get-pip.py`

#### Environment setup
- `cd ~/ && mkdir git && cd ~/git/`
- `git clone https://github.com/marcelovca90/DS-balancing-dataset.git`
- `cd DS-balancing-dataset`
- `chmod +x run.sh`
- `./run.sh | tee run.log`
