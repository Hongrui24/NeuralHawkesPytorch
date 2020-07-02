# Neural Hawkes Pytorch
This repository is a more concise and simpler pytorch implementation of the model in paper Hongyuan Mei, Jason Eisner [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)

## Introduction:
## Model Description:
The model is highly based on LSTM. This [website](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21) provides a quick introduction to a typical LSTM model and background knowledge in Deep Learning. 
#### Model 
The following is a recap of the paper by Hongyuan Mei:<br>
We use continuous-time LSTM, a type of recurrent network to model the Hawkes Process. The model is called continuous because the memory cell c will decay exponentially to some constant value c̅ with rate <img src="https://render.githubusercontent.com/render/math?math=\delta">, and the value of cell c and c̅ will be updated once the model get input (k<sub>i</sub>, t<sub>i</sub>).<br>
Because in the model's decaying architecture, it calculates the inter-event duration. Thus, the input to the model will be (k<sub>i</sub>, d<sub>i</sub>) instead, where d<sub>i</sub> = t<sub>i</sub> - t<sub>i-1</sub>, and d<sub>1</sub> = t<sub>1</sub>. Before input the event sequence into our model, we will pad an event <k<sub>0</sub>, d<sub>0</sub>> = <k, 0> to start the sequence for initial value of LSTM to the sequence. For each LSTM cell, we first embed the event type k<sub>i</sub> into a vector K<sub>i</sub> of size k+1, and feed the vector into the LSTM cell. The cell is updated by the following equations:
<pre><img src="https://render.githubusercontent.com/render/math?math=i_{{\imath}+1} = sigmoid(W_ik_{\imath} \oplus U_ih(t_{\imath}) \oplus d_i)">
<img src="https://render.githubusercontent.com/render/math?math=f_{{\imath}+1} = sigmoid(W_fk_{\imath} \oplus U_fh(t_{\imath}) \oplus d_f)">
<img src="https://render.githubusercontent.com/render/math?math=z_{{\imath}+1} = tanh(W_zk_{\imath} \oplus U_zh(t_{\imath}) \oplus d_z)">
<img src="https://render.githubusercontent.com/render/math?math=o_{{\imath}+1} = sigmoid(W_ok_{\imath} \oplus U_oh(t_{\imath}) \oplus d_o)">
<img src="https://render.githubusercontent.com/render/math?math=c_{{\imath}+1} = f_{i+1} \times c(t_i) \oplus i_{\imath+1} \times z_{\imath+1}">
<img src="https://render.githubusercontent.com/render/math?math=\bar{c_{\imath+1}} =  \bar{f_{\imath+1}} \times \bar{c_i} \oplus \bar{i_{\imath+1}} \times z_{\imath+1}">
<img src="https://render.githubusercontent.com/render/math?math=\delta_{\imath+1} = softplus(W_dk_{\imath} \oplus U_dh(t_i) \oplus d_d)"></pre>

#### Prediction on Time (duration) and types:
As described in the paper once we have calculated the intensity function λ(t) for the time t in the future, the density function at time t is given by :
<pre><img src="https://render.githubusercontent.com/render/math?math=p(t) = \lambda (t) * e ^ {\int_{t_{i-1}}^{t} \lambda (s)ds}"></pre>
We can calculate when will the next event happens by the expectation of probability density function on time:
<pre><img src="https://render.githubusercontent.com/render/math?math=t_{estimated} = \int_{t_{i-1}}^{\infty} tp(t) dt = \int_{t_{i-1}}^{\infty} t\lambda(t) * e ^ {\int_{t_{i-1}}^{t} \lambda (s)ds}dt"></pre>
The paper states that the integrals above can be estimated by Monte Carlo Simulation, but it does not give a clear algorithm for that simulation. When we think intuitively, we may sample N event times which happen after t<sub>i-1</sub> to calculate the integral in the expectation, and for each event time sampled we may sample another M event times between t<sub>i-1</sub> and the event time which is sampled before to calculate the integral in the exponential term. If we calculate estimated time for the next event in this way, not only will it takes a long time to calculate, but also the accuracy may decrease when M is small for large t with fixed M. Thus we purpose an algorithm below<br><br>

Since our model takes in inter-event duration as input, we will estimate the inter-event duration (Δt<sub>estimated</sub>) to the next future time by the equation:
<pre><img src="https://render.githubusercontent.com/render/math?math=\Delta t_{estimated} = \int_{0}^{\infty} \Delta tp(\Delta t) d \Delta t = \int_{0}^{\infty} \Delta t\lambda(\Delta t) * e ^ {\int_{0}^{\Delta t} \lambda (s)ds} d \Delta t"> where Δt = t - t<sub>i-1</sub> </pre>
We can get t<sub>estimated</sub> by adding t<sub>i-1</sub> to Δt<sub>estimated</sub><br><br>
First we simulate N inter-event durations, where N is set to be 10000 by default. By technique difficulty of representing infinity, we assume that 20 times the max duration in history is a good representation for infinity:
<pre><img src="https://render.githubusercontent.com/render/math?math=0 < \Delta t_1 < \Delta t_2 < \Delta t_3 < ...... < \Delta t_N < 20 \times MaxDuratiom "></pre>
We use the same simulated points for both integrals. We first expanded the estimated sequence to N sequences of the same length and value plus a simulated time (duration) in order accordingly. We can then calculate the corresponding value of c(Δt<sub>k</sub>), h(Δt<sub>k</sub>), and λ(Δt<sub>k</sub>) by Neural Hawkes model in one run. We store all these λ(Δt<sub>k</sub>) in a list for latter use. Let m be an integer that is less than or equal to N. It is reasonable to think that Δt<sub>1</sub>, Δt<sub>2</sub>, ... Δt<sub>m</sub> are uniformly simulated in range [0, Δt<sub>m</sub>], and we can assume that Δt<sub>1</sub> is really small, duration between 2 consecutive simulations are small. Thus, we can estimate the integral on exponential term by:
<pre><img src="https://render.githubusercontent.com/render/math?math=\int_{0}^{\Delta t_m}\lambda (s) ds \approx \frac {t_m}{m}\sum_{k=1}^{k=m}\lambda (\Delta t_k)">, and <img src="https://render.githubusercontent.com/render/math?math=p(\Delta t_m) = \lambda (\Delta t_k) \times e ^ {\int_{0}^{\Delta t_m}\lambda (s) ds} \approx  \lambda (\Delta t_k) \times e ^ {\frac {t_m}{m}\sum_{k=1}^{k=m}\lambda (\Delta t_k)}"></pre>
Then, we can estimate inter-event time to the next event by:
<pre><img src="https://render.githubusercontent.com/render/math?math=\Delta t_{estimated} = \int_{0}^{\infty} \Delta tp(\Delta t) dt \approx  \frac {20 \times MaxDuration}{N}  \sum_{m=1}^{m=N} \Delta t_m p(\Delta t_m) "></pre>
<br>
By this method, we do not need to simulate N * M points to estimate two integrals above as thought intuitively. The algorithm also has some kinds of accuracy as the test results looks good.  

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

4. Google Colab:
Because of the complexity of the model, and long training time, it is better to train and test the model on cloud such as Google Colab rather than train the model on your laptop or desktop. The purpose of using Google Colab is to accelerate the training process and protect your personal laptop from overheated caused by a long time intense computing. If you are using a desktop built for neural network training or scientific computing, you may simply ingore this section. <br><br>
Google Colab is a plotform which allow you to write and run python Jupyter Notebook with CPU, GPU or TPU that are designed for Neural Network training. Google Colab also allows you to type linux command line to execute python files such as files in this repository. <br><br>
To use the Google Colab, you must use the chrome browser, log in to your google account, and follow the picture below:
![colab1](https://user-images.githubusercontent.com/54515153/86067942-16b93380-ba44-11ea-9a3c-19393b3eed95.JPG)
![colab2](https://user-images.githubusercontent.com/54515153/86068280-ede56e00-ba44-11ea-9ace-867ae3ae6ac4.JPG)
It is recommanded to use GPU to train the model. To change to GPU mode, select Runtime, Change run time type, and in Hardware accelerator select GPU.
Type the commands blow cell by cell:
<pre>!git clone  https://github.com/Hongrui24/NeuralHawkesPytorch</pre>
<pre>!cd NeuralHawkesPytorch</pre>
Then you can type the command in this section 2. and 3. to train and test the model. 


## Testing Results:
1. The first test is training the model with data named "conttime" described in Hongyuan Mei's Paper with lr = 0.01, epochs = 30, mini batch size = 10. The log likelihood (not negative log likelihood) during the training has the plot ![log-likelihood-graph](https://user-images.githubusercontent.com/54515153/85951273-1af42c80-b930-11ea-8193-9bade5181951.jpg)
<bre />
When we test our trained model with the test file named test.pkl, we get log-likelihood over the seqs is -0.99, log-likelihood over the type is -1.44, and log-likelihood over the time is 0.447. All these numbers fit the range described inside the paper section C2, where log-likelihood over seqs, type and time should be -1.00 to -0.98, -1.44 to -1.43, and -0.440 to 0.455 accordingly. This test shows that the model we built is the model described inside the paper.
<br/><br/>

2. We also test out model with data provided in Du, Nan, et al. [“Recurrent Marked Temporal Point Processes.”](https://www.kdd.org/kdd2016/subtopic/view/recurrent-temporal-point-process) paper about self-correcting and hawkes. We train the model for 10 epochs with learning rate = 0.01 and truncated sequence length = 75. (Picture stack below from left to right, from top to bottom are result by Neural Hawkes, RMTPP, and Du Nan's Paper)<br />
- Result of hawkes with comparison to the result in [pytorch implementation](https://github.com/Hongrui24/RMTPP-pytorch) and results in Du Nan's paper as below:
<p float="left">
<img src='https://user-images.githubusercontent.com/54515153/86301854-f7054500-bbd4-11ea-8f04-634b6704514d.png' width='450' height='350'> <img src = 'https://user-images.githubusercontent.com/54515153/86301951-3e8bd100-bbd5-11ea-94c3-40cb083b7c63.png' width='450' height='350'>
</p>
<img src = 'https://user-images.githubusercontent.com/54515153/85952682-05373500-b939-11ea-8224-b5d174d7cbf4.JPG'>



- Result of self-correcting with comparison to result in [pytorch implementation](https://github.com/Hongrui24/RMTPP-pytorch) and results in Du Nan's paper as below:
<p float="left">
  <img src='https://user-images.githubusercontent.com/54515153/86302009-709d3300-bbd5-11ea-816e-7acdc8b3a013.png' width='450' height='350'><img src='https://user-images.githubusercontent.com/54515153/86302085-a80bdf80-bbd5-11ea-9371-802347398c6f.png' width='450' height='350'>
 </p>
 <img src='https://user-images.githubusercontent.com/54515153/85622697-3d86fc80-b635-11ea-99f1-d6d5835e642c.JPG'>
 
We find that the models trained by Neural Hawkes have a better prediction on the event duration and intensity comparing to model trained by RMTPP. What's more the prediction result by the model trained by Neural Hawkes is close to the prediction result by optimal estimators whose equations are known. 

## Acknowledgement:
This model is built by Hongrui Lyu, supervised by Hyunouk Ko and Dr. Huo. The file cont-time-cell is just a copy from Hongyuan Mei's code, but all other files are written by us. As notice by the original github page of pytorch implementation, this [license](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing/blob/master/LICENSE) need to be included. 
