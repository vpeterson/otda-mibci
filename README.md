# DOMAIN ADAPTATION BASED ON OPTIMAL TRANSPORT FOR MI-BCI

You will find here all the codes and instructions you need in order to reproduce the experiments performed in "Transfer Learning based on Optimal Transport for Motor Imagery Brain-Computer Interfaces", by Victoria Peterson, Nicolás Nieto, Dominik Wyser, Olivier Lambercy, Roger Gassert, Diego H. Milone and Ruben D. Spies

In MIOTDAfunctions.py you will find all the defined functions implemented for learning the mapping, the training subset as well as choosing the best regularization parameters. 

Two notebook examples are provided, each one for each online testing scenario considered in our study. In addition, a basic example on how OTDA can be used in MI-BCI is provided.  
## Installation guidelines
This guidelines are based on [Anaconda](https://www.anaconda.com/distribution/) distribution.
The library has been tested on Linux and Windows.

## Install requirements for reproducing the experiments
If you want to reproduce experiments and results, where OTDA and data alignment methods are implemented follow these steps:
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
5. Download or clone [OTDA-MIBCI](https://github.com/vpeterson/otda-mibci.git)
6. Go to the OTDA-MIBCI downloaded directory
7. Install Jupyter (if needed)
```
 pip3 install jupyter
```
8. Run the examples

#### Blockwise test
```
jupyter notebook paper_example_blockwise.ipynb
```
#### Samplewise test
```
jupyter notebook paper_example_samplewise.ipynb
```
## Install requirements for using OTDA-MIBCI
If you just want to use OTDA, and learn how to use it, follow these steps:
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
4. Download or clone [OTDA-MIBCI](https://github.com/vpeterson/otda-mibci.git)
5. Go to the OTDA-MIBCI downloaded directory
6. Run the basic example
```
jupyter notebook Basic_example.ipynb
```
##### And you are ready. Happy coding!

