## DOMAIN ADAPTATION BASED ON OPTIMAL TRANSPORT FOR MI-BCI

You will find here all the codes and instructions you need in order to reproduce the experiments performed in "Transfer Learning based on Optimal Transport for Motor Imagery Brain-Computer Interfaces", by Victoria Peterson, Nicolás Nieto, Dominik Wyser, Olivier Lambercy, Roger Gassert, Diego H. Milone and Ruben D. Spies

In MIOTDAfunctions.py you will find all the defined functions implemented for learning the mapping, the training subset as well as choosing the best regularization parameters. 

Two notebook examples are provided, each one for each online testing scenario considered in our study. In addition, a basic example on how OTDA can be used in MI-BCI is provided.  
## Requirements
Python 3.7:
1) [MNE](https://mne.tools/stable/index.html)
2) [POT](https://github.com/PythonOT/POT)
3) [Scikit Learn](https://scikit-learn.org/stable/) 
4) [RPA](https://github.com/plcrodrigues/RPA)

## Installation guidelines
This guidelines are based on [Anaconda](https://www.anaconda.com/distribution/) distribution.
The library has been tested on Linux and Windows.

If you want to reproduce experiments and results, where OTDA and data alignment methods are implemented follow these steps:
#### Install requirements for reproduce the experiments
1. Create conda environment
```
conda env create -f environment_paper.yml

```
2. Activate conda environment
```
conda activate test_otdapaper
```
3. Download or clone the [RPA](https://github.com/plcrodrigues/RPA)
 repository
4. Go to the downloaded directory and run
```
python setup.py develop
```
5. Install Jupyter (if needed)
```
 pip3 install jupyter
```

If you just want to use OTDA and run the basic example follow these steps:
#### Install requirements for OTDA
1. Create conda environment
```
conda env create -f environment_otda.yml

```
2. Activate conda environment
```
conda activate otda
```
3. Install Jupyter (if needed)
```
 pip3 install jupyter
```

#### Install OTDA-MIBCI
Option 1: using git
```
 git clone https://github.com/vpeterson/otda-mibci.git
```
Option 2: download this folder and unzip it
#### Examples:
##### Blockwise OTDA test
Locate where otda-mibci folder is:
```
cd otda-mibci
```
Run the notebook:
```
jupyter notebook Example_blockwise_MIOTDA.ipynb
```
##### Samplewise OTDA test
If located on the otda-mibci folder run the notebook:
```
jupyter notebook Example_blockwise_MIOTDA.ipynb
```
##### And you are ready. Happy coding!

