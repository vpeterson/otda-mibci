# DOMAIN ADAPTATION BASED ON OPTIMAL TRANSPORT FOR MI-BCI

You will find here all the codes and instructions needed to reproduce the experiments performed in "Transfer Learning based on Optimal Transport for Motor Imagery Brain-Computer Interfaces", by Victoria Peterson, Nicolás Nieto, Dominik Wyser, Olivier Lambercy, Roger Gassert, Diego H. Milone and Ruben D. Spies

In MIOTDAfunctions.py you will find all the defined functions implemented for learning the mapping, the training subset as well as choosing the best regularization parameters. 

Two notebook examples are provided, each one for each online testing scenario considered in our study. In addition, a basic example on how OTDA can be used in MI-BCI is provided.  
## Installation guidelines
This guidelines are based on [Anaconda](https://www.anaconda.com/distribution/) distribution.
The library has been tested on Linux and Windows.

### Install requirements for reproducing the experiments
If you want to reproduce experiments and results, where OTDA and data alignment methods are implemented follow these steps:
1. Donwload and extract the zip or clone [OTDA-MIBCI](https://github.com/vpeterson/otda-mibci.git)
2. Go to the OTDA-MIBCI directory
3. Create conda environment
```
conda env create -f environment_paper.yml
```
4. Activate conda environment
```
conda activate test_otdapaper
```
5. Download and extract the zip or clone the [RPA](https://github.com/plcrodrigues/RPA)
 repository
6. Go to the downloaded directory and run
```
python setup.py develop
```
7. Go again to the OTDA-MIBCI downloaded directory
8. Run the examples
**Block-wise test**
```
jupyter notebook paper_example_blockwise.ipynb
```
**Sample-wise test**
```
jupyter notebook paper_example_samplewise.ipynb
```
## Install requirements for using OTDA-MIBCI
If you just want to use OTDA, and learn how to use it, follow these steps:
1. Donwload and extract the zip or clone [OTDA-MIBCI](https://github.com/vpeterson/otda-mibci.git)
2. Go to the OTDA-MIBCI directory
3. Create conda environment
```
conda env create -f environment_otda.yml
```
4. Activate conda environment
```
conda activate otda
```
5. Run the basic example
```
jupyter notebook Basic_example.ipynb
```
##### And you are ready. Happy coding!

