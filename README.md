## DOMAIN ADAPTATION BASED ON OPTIMAL TRANSPORT FOR MI-BCI

You will find here all the codes and instructions you need in order to reproduce the experiments performed in "Transfer Learning Based on Optimal Transport for MI-BCI", by Victoria Peterson, Diego H. Milone, Dominik Wyser, Olivier Lambercy, Roger Gassert and Ruben Spies.

In MIOTDAfunctions.py you will find all the defined functions implemented for learning the mapping, the training subset as well as choosing the best regularization parameters. 

Two notebook examples are provided, each one for each online testing scenario considered in our study. 

## Requirements 
#### (Python 3)
1) MNE (https://mne.tools/stable/index.html)
2) POT (https://github.com/PythonOT/POT)
3) Scikit Learn (https://scikit-learn.org/stable/)

## Installation guidelines (based on Anaconda Distribution)
### create conda environment
$conda create --name otda pip
### Activate conda environment
$conda activate otda
### Install sklearn
$pip3 install sklearn
### Install Jupyter
$pip3 install jupyter
### Install MNE
$pip3 install mne
### Install POT 
$conda install -c conda-forge pot
### Install OTDA
#### opt. 1: using git
$git clone https://github.com/vpeterson/otda-mibci.git
#### opt. 2: download this folder and unzip it
### Run example blockwise OTDA test
#### locate where otda-mibci folder is:
$cd otda-mibci
#### run the notebook:
$jupyter notebook Example_blockwise_MIOTDA.ipynb
### Run example samplewise OTDA test
#### if located on the otda-mibci folder run the notebook:
$jupyter notebook Example_blockwise_MIOTDA.ipynb

### And you are ready. Happy coding!

