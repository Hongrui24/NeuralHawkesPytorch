# Neural Hawkes Pytorch

This repository is a simple pytorch implementation of the paper Hongyuan Mei, Jason Eisner [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)
<br />
This repository is under development now. The model can only be trained, and be tested on log likelihood on seqs, time, and type in the current version. More functions are comming soon.
<br />

In order to train the model, please use 
<pre>!python train.py --help</pre> for more information

In order to test the model, please type 
<pre>!python test.py --help</pre> for more information.

<br />Some of the data files are too large to be push to this repository. These are the data used originally in Hongyuan Mei et al's paper. Please refer to [this](https://github.com/HMEIatJHU/neurawkes/tree/master/data) page to get the data. The direction of the data file should be parallel with the train and test file with name "data". Inside the "data" folder should be the folder named for each test like "conttime", "data_hawkes" etc. Each test folder should contain "train.pkl", and "test.pkl". We use "conttime" as folder name for "data_conttime", "data_hawkes" for "data_hawkes" and "data_hawkeshib" for "data_hawkesinhib" for the paper's data. 


As notice by the original github page of pytorch implementation, this [license](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing/blob/master/LICENSE) need to be included. 
