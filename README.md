# Neural Hawkes Pytorch
This repository is a more concise and simpler pytorch implementation of the model in paper Hongyuan Mei, Jason Eisner [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)

## Introduction:
## Model Description:

#### Prediction on Time (duration) and types:
As described in the paper, the density function is given by (assume that we predict duration: time to the next event):
<pre> p(t<sub>m</sub>) = λ(t<sub>m</sub>)exp(-integral<sub>0</sub><sup>t<sub>m</sub></sup>λ(s)ds), where m < N. N is the number of simulated points </pre>
The predicted time is estimated by expectation on density function. The paper states that the integrals above can be estimated by Monte Carlo Simulation, but it does not give a clear algorithm for that simulation. Thus we purpose an algorithm below<br><br>
First we simulate N events, where N is set to be 10000 by default. By technique difficulty of representing infinity, we assume that 20 times the max duration in the previous known data is a good representation for infinity. We first simulated N points uniformly in 0 to 20*max_duration:
<pre> 0 < t<sub>1</sub> < t<sub>2</sub> < t<sub>3</sub> < .... < t<sub>N</sub> < 20*max_duration</pre>
We use the same simulated points for both integrals. We first expanded the estimated sequence to N sequences of the same length and value plus a simulated time (duration) in order for N sequences. We can then calculate the corresponding estimation of c(t<sub>k</sub>), h(<sub>t</sub>), and λ(<sub>t</sub>) for each sequence. It is reasonable to think that t<sub>1</sub>, t<sub>2</sub>, ... t<sub>m</sub> are uniformly simulated in range [0, <sub>t</sub>], and we can assume that t1 is really small, and duration between 2 consecutive simulations are small. Thus, we can estimate the first integral in simulation by:
<pre> integral<sub>0</sub><sup>t<sub>m</sub></sup>λ(s)ds = t<sub>m</sub> / m * (summation<sub>k=1</sub><sup>k=m</sup>{λ(t<sub>k</sub>}) </pre>
Thus we can get an estimated density function for each simulated points. Then, we can estimate the time by:
<pre> t<sub>est</sub> = 20*max_duration / N * (summation<sub>m=1</sub><sup>m=N</sup>{t<sub>m</sub>p(t<sub>m</sub>)}) </pre>
<br>
Then we can use the predicted time to calculate estimated intensity and types by inputting the estimated time into the original sequence to find <pre>λ(t<sub>est</sub>)</pre> and find estimated types by <pre>type = max(λ<sub>i</sub>(t<sub>est</sub>)) for i types of event. </pre>
<br>
By this method, we do not need to simulate N * N points to estimate two integrals above as thought intuitively. We can estimate one point in O(N) times, and the result are pretty accurate as shown in the test results section. 

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
<br/><br/>

2. We also test out model with data provided in Du, Nan, et al. [“Recurrent Marked Temporal Point Processes.”](https://www.kdd.org/kdd2016/subtopic/view/recurrent-temporal-point-process) paper about self-correcting and hawkes. We train the model for 10 epochs with learning rate = 0.01 and truncated sequence length = 75. (Picture stack below from left to right, from top to bottom are result by Neural Hawkes, RMTPP, and Du Nan's Paper)<br />
- Result of hawkes with comparison to the result in [pytorch implementation](https://github.com/Hongrui24/RMTPP-pytorch) and results in Du Nan's paper as below:
<p float="left">
<img src='https://user-images.githubusercontent.com/54515153/85952673-e173ef00-b938-11ea-8d1f-67ed3e67ec00.png' width='460' height='350'> <img src = 'https://user-images.githubusercontent.com/54515153/84570792-a630c800-ad5d-11ea-972e-a809f0865add.png' width='460' height='350'>
</p>
<img src = 'https://user-images.githubusercontent.com/54515153/85952682-05373500-b939-11ea-8224-b5d174d7cbf4.JPG'>



- Result of self-correcting with comparison to result in [pytorch implementation](https://github.com/Hongrui24/RMTPP-pytorch) and results in Du Nan's paper as below:
<p float="left">
  <img src='https://user-images.githubusercontent.com/54515153/85962642-c5e00700-b97f-11ea-9c6a-d117d8329350.png' width='460' height='350'><img src='https://user-images.githubusercontent.com/54515153/84570795-a9c44f00-ad5d-11ea-9b71-30632793f9b4.png' width='460' height='350'>
 </p>
 <img src='https://user-images.githubusercontent.com/54515153/85622697-3d86fc80-b635-11ea-99f1-d6d5835e642c.JPG'>
 
We find that the models trained by Neural Hawkes have a better prediction on the event duration and intensity comparing to model trained by RMTPP. What's more the prediction result by the model trained by Neural Hawkes is close to the prediction result by optimal estimators whose equations are known. 

## Acknowledgement:
This model is built by Hongrui Lyu, supervised by Hyunouk Ko and Dr. Huo. The file cont-time-cell is just a copy from Hongyuan Mei's code, but all other files are written by us. As notice by the original github page of pytorch implementation, this [license](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing/blob/master/LICENSE) need to be included. 
