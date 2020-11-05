## DOMAIN ADAPTATION BASED ON OPTIMAL TRANSPORT FOR MI-BCI

You will find here all the codes and instructions you need in order to reproduce the experiments performed in "Transfer Learning based on Optimal Transport for Motor Imagery Brain-Computer Interfaces", by Victoria Peterson, et al.

In MIOTDAfunctions.py you will find all the defined functions implemented for learning the mapping, the training subset as well as choosing the best regularization parameters. 

Two notebook examples are provided, each one for each online testing scenario considered in our study. 
## Requirements 
Python 3:
1) [MNE](https://mne.tools/stable/index.html)
2) [POT](https://github.com/PythonOT/POT)
3) [Scikit Learn](https://scikit-learn.org/stable/)
## Installation guidelines
This guidelines are based on [Anaconda](https://www.anaconda.com/distribution/) distribution.
The library has been tested on Linux and Windows.
#### Install requirements
1. Create conda environment
```
conda create --name otda pip
```
2. Activate conda environment
```
conda activate otda
```
3. Install Sklearn
```
 pip3 install sklearn
```
4. Install Jupyter (if needed)
```
 pip3 install jupyter
```
5. Install MNE (more information [here](https://mne.tools/stable/install/mne_python.html))
```
 pip3 install mne
```
6. Install POT (more information [here](https://pythonot.github.io/))
```
 conda install -c conda-forge pot
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

