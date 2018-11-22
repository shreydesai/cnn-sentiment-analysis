#!/bin/bash

# install python 3.5
sudo apt-get update
sudo dpkg --configure -a
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.5

# install virtual environment
sudo apt install python-virtualenv

# create virtual environment
virtualenv -p python3.5 venv
source venv/bin/activate
cat requirements.txt | xargs pip install
