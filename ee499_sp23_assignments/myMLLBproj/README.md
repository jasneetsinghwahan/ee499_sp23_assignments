### file description
file named 'distil_hat.py' used for distilling the data
-- requires a folder named 'artrawdata' be present in the current wording directory and shall output the results of the distillation to a file with name beginning with 'op-*'
-- made by me

file named 'artweights.h5' has the weights of the model that are copied from the repo: https://github.com/Keitokuch/linux-4.15-lb commit ID 3f2f09b38210a70ef44017e427f256439a0fbdec

file named 'genrawtraindata.py' is a modified version of original file that is part of the paper [*Machine Learning for Load Balancing in the Linux Kernel*](https://doi.org/10.1145/3409963.3410492)

file named 'traintfv.py' is the training code file, part of which is copied (not exactly) from the train.py file that was part of the paper and available at https://github.com/keitokuch/MLLB
-- results are stored in two folders named 'artresults' and 'artmodel'
-- accepts command-line argyuments but provide defaults and morover command-line arguments are set to 'not required/optional'

file named 'training_config.py' has the paramaters to be extracted for generating the processed data
-- originally copied from the trainining_config.py file that was part of the paper and available at https://github.com/keitokuch/MLLB

file named 'prep.py' is a modified version of original file that is part of the paper
-- it reads the raw data as fetched from the kernel from the folder titled 'rawdata' and stores the preprocessed data in the folder titled 'procdata'

file named 'dump_lb.py' copied from the repo https://github.com/keitokuch/MLLB and is used to generate the data from the kernel

## folder description
folder titled 'rawdata' has the raw data extracted from the modified linux kernel


## instructions for reproducing the results
python3 distil_hat.py               # generate the temporary data to work upon
python3 traintfv.py                 # train the model and store the results

## instructions (not used) since I could only fetch  6/1 fields
python3 dump_lb.py                  # generates the data, but remember to provdie tag each time you run this file
python3 prep.py                     # pre-process the data
python3 traintfv.py                 # train the model and store the results