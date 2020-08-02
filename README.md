## DOMAIN ADAPTATION BASED ON OPTIMAL TRANSPORT FOR MI-BCI

You will find here all the codes and instructions you need in order to reproduce the experiments performed in "Transfer Learning Based on Optimal Transport for MI-BCI", by Victoria Peterson, Diego H. Milone, Dominik Wyser, Olivier Lambercy, Roger Gassert and Ruben Spies.

In MIOTDAfunctions.py you will find all the defined functions implemented for learning the mapping, the training subset as well as choosing the best regularization parameters. 

Two notebook examples are provided, each one for each online testing scenario considered in our study. 

## Requirements 
### (Python 3)
# MNE
# POT
# SKLEARN

## Installation guidelines (based on Anaconda Distribution)
### 1) create conda environment
conda create --name otda pip
### 2) Activate conda environment
conda activate otda
### 3) Install sklearn
pip3 install sklearn
### 4) Install Jupyter
pip3 install jupyter
### 5) install MNE (more information here https://mne.tools/stable/index.html)
pip3 install mne
### install POT (more information here ttps://github.com/PythonOT/POT)
conda install -c conda-forge pot
### install OTDA
git clone https://github.com/vpeterson/otda-mibci.git
### blockwise OTDA test
cd otda-mibci
jupyter notebook Example_blockwise_MIOTDA.ipynb
