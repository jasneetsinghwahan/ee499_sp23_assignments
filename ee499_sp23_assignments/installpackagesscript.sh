#!/usr/bin/bash

conda install numpy
conda install pandas
conda install seaborn
conda install matplotlib
conda install scikit-learn
python -m pip install jupyter notebook -U
conda install -n ML ipykernel --update-deps --force-reinstall
mkdir generatedData
conda install openpyxl
conda install -c conda-forge xlsxwriter
conda install -c anaconda h5py
conda install -c conda-forge autograd
conda install -c pytorch pytorch
pip install torchvision
pip install opencv-python
pip install imutils
pip install chardet
pip install charset-normalizer
