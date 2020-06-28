# Neural Hawkes Pytorch
This repository is a more concise and simpler pytorch implementation of the model in paper Hongyuan Mei, Jason Eisner [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)

## Introduction:
## Model Description:
## Train and Testing the Model:
1. To run the program on your computer, please make sure that you have the following files and packages being downloaded.<br />
- Python3: you can download through the link here: https://www.python.org/ </pre>
- Numpy: you can dowload it through command <pre>pip install numpy</pre>
- Scikit-Learn: you can download it through command <pre>pip install sklearn</pre>
- matplotlib: you can download it through command <pre>pip install matplotlib</pre>
- pytorch installation is more complicated than the package described above. You can go to https://pytorch.org/get-started/locally/ for more information. If you still cannot install it on windows computer through pip, you can download Anaconda first and then download the pytorch through method described here: https://dziganto.github.io/data%20science/python/anaconda/Creating-Conda-Environments/ <br /><br />
<br />
2. In order to train the model, please type the command below for more information:
<pre>!python train.py --help</pre>
Examples include:
<pre>!python train.py --dataset conttime</pre><pre>!python train.py --dataset hawkes --seq_len 75 --batch_size 64</pre>
<br />
3. In order to test the model, please type the command below for more information:
<pre>!python test.py --help</pre>
Examples include:
<pre>!python test.py --dataset conttime --test_type 2</pre><pre>!python test.py --dataset self-correcting --test_type 1</pre>

<br />Some of the data files are too large to be push to this repository. These are the data used originally in Hongyuan Mei et al's paper. Please refer to [this](https://github.com/HMEIatJHU/neurawkes/tree/master/data) page to get the data. The direction of the data file should be parallel with the train and test file with name "data". Inside the "data" folder should be the folder named for each test like "conttime", "data_hawkes" etc. Each test folder should contain "train.pkl", and "test.pkl". We use "conttime" as folder name for "data_conttime", "data_hawkes" for "data_hawkes" and "data_hawkeshib" for "data_hawkesinhib" for the paper's data. 

## Testing Results:
1. The first test is training the model with data named "conttime" described in Hongyuan Mei's Paper with lr = 0.01, epochs = 30, mini batch size = 10. The log likelihood (not negative log likelihood) during the training has the plot ![log-likelihood-graph](https://user-images.githubusercontent.com/54515153/85951273-1af42c80-b930-11ea-8193-9bade5181951.jpg)
<bre />
When we test our trained model with the test file named test.pkl, we get log-likelihood over the seqs is -0.99, log-likelihood over the type is -1.44, and log-likelihood over the time is 0.447. All these numbers fit the range described inside the paper section C2, where log-likelihood over seqs, type and time should be -1.00 to -0.98, -1.44 to -1.43, and -0.440 to 0.455 accordingly. This test shows that the model we built is the model described inside the paper.
<br/><br /><br />
2. We also test out model with data provided in Du, Nan, et al. [“Recurrent Marked Temporal Point Processes.”](https://www.kdd.org/kdd2016/subtopic/view/recurrent-temporal-point-process) paper about self-correcting and hawkes. We train the model for 10 epochs with learning rate = 0.01 and truncated sequence length = 75.<br />
hawkes: We get the result in the following plot <br />
![result](https://user-images.githubusercontent.com/54515153/85952355-aec8f700-b936-11ea-8639-d2e9b00e5482.png)        

## Acknowledgement:
As notice by the original github page of pytorch implementation, this [license](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing/blob/master/LICENSE) need to be included. 
