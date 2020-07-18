# Neural Hawkes Pytorch
This repository is a more concise and simpler pytorch implementation of the model in paper Hongyuan Mei, Jason Eisner [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)

## Introduction:
A sequence of events with different types are often generated in our lives. For instance, a patient may be diagnosed with different diseases in his or her record history; a kind of stock may be sold or bought several times in a given day. We can define that the i<sup>th</sup> event in such a sequence above is a tuple (k<sub>i</sub>, t<sub>i</sub>) where k<sub>i</sub> denote the type of the event and t<sub>i</sub> denote when does this event happens. Therefore, a sequence of events can be represented in a sequence of such tuples above. Such sequences are usually called Marked Point Process or Multivariate Point Process. The problem we care about is to predict when will the next event happens and what will be the event type given a stream of events.That is given a stream of event of form:
<pre>(k<sub>1</sub>, t<sub>1</sub>), (k<sub>2</sub>, t<sub>2</sub>), (k<sub>3</sub>, t<sub>3</sub>) ... (k<sub>n</sub>, t<sub>n</sub>)</pre>
we want to predict the next event time and type (k<sub>n+1</sub>, t<sub>n+1</sub>)<br>
## Previous Work and Background Knowledge

### Intensity Function in Point Process
![Intensity1](https://user-images.githubusercontent.com/54515153/87855728-2fe92d80-c8e8-11ea-818a-758e1b84f06f.JPG)
![Intensity2](https://user-images.githubusercontent.com/54515153/87831532-54c4ad00-c8b6-11ea-82c2-6ad6b43a4cca.JPG)<br>
Please refer to [J. G. Rasmussen. Temporal point processes: the
conditional intensity function. 2009.](https://arxiv.org/abs/1806.00221) for proof and detailed math formular deductions in the place with mark [1] above.

### Hawkes Process
![Hawkes](https://user-images.githubusercontent.com/54515153/87855731-32e41e00-c8e8-11ea-89a4-00f1711508f9.JPG)

### LSTM
![LSTM1](https://user-images.githubusercontent.com/54515153/87856492-991f6f80-c8ed-11ea-9bae-362e3a0c3fa2.JPG)
![LSTM2](https://user-images.githubusercontent.com/54515153/87855736-37103b80-c8e8-11ea-8196-516a4e644b82.JPG)
![LSTM3](https://user-images.githubusercontent.com/54515153/87855831-e0573180-c8e8-11ea-933e-2e5829cafda4.JPG)<br>
To learn more about how Neural Network, RNN an LSTM works, [Dive into Deep Learning](https://d2l.ai/) is a good source. 

## Model Description:

### Model and Model Training
![training1](https://user-images.githubusercontent.com/54515153/87862402-bb30e600-c91d-11ea-867c-1d82f90ff361.JPG)
![training2](https://user-images.githubusercontent.com/54515153/87862403-bd934000-c91d-11ea-85b2-5aa80093b448.JPG)
![training3](https://user-images.githubusercontent.com/54515153/87862406-bec46d00-c91d-11ea-9246-a1a0edef354e.JPG)
![training4](https://user-images.githubusercontent.com/54515153/87862407-c08e3080-c91d-11ea-83cb-c99e1cbc090e.JPG)
![training5](https://user-images.githubusercontent.com/54515153/87862408-c1bf5d80-c91d-11ea-9a79-61e9eb1567f3.JPG)

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

5. Notice:
The dataset in this repository is truncated from the original data due to the large dataset and long training time. You may train the data in this repository with at least 20 epochs to get a similar result below. The training process takes about 30 minutes, and the original dataset may be trained for at least 4 hours for about 10 epochs. The original dataset can be found in [this page](https://github.com/dunan/NeuralPointProcess/tree/master/data/synthetic) and [here](https://github.com/HMEIatJHU/neurawkes/tree/master/data)
## Testing Results:
1. The first test is training the model with data named "conttime" described in Hongyuan Mei's Paper with lr = 0.01, epochs = 30, mini batch size = 10. The log likelihood (not negative log likelihood) during the training has the plot ![log-likelihood-graph](https://user-images.githubusercontent.com/54515153/85951273-1af42c80-b930-11ea-8193-9bade5181951.jpg)
<bre />
Test results:
<table>
  <tr>
    <th></th>
    <th>Model Result</th>
    <th>Result on Paper</th>
  </tr>
  <tr>
    <td>log-likelihood over seqs</td>
    <td>-0.99</td>
    <td>-1.00 to -0.98</td>
  </tr>
  <tr>
    <td>log-likelihood over time</td>
    <td>0.447</td>
    <td>0.440 to 0.455</td>
  </tr>
  <tr>
    <td>lo-likelihood over type</td>
    <td>-1.44</td>
    <td>-1.44 to -1.43</td>
 </table>

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

## Data Source and Structure:
We use the data provided by the Hongyuan Mei and Du Nan to do tests. 
<table>
  <tr>
    <th>Name</th>
    <th>Number of types</th>
    <th>Number of training sequence</th>
    <th>Number of testing sequence</th>
    <th>Sequence Length Mean</th>
    <th>Sequence Length Min</th>
    <th>Sequence Length Max</th>
  </tr>
  <tr>
    <td>data_hawkes, data_hawkeshib, data_conttime</td>
    <td>5</td>
    <td>8000</td>
    <td>1000</td>
    <td>60</td>
    <td>20</td>
    <td>100</td>
  </tr>
  <tr>
    <td>MIMIC-II(1)(2)(3)(4)(5)</td>
    <td>75</td>
    <td>527</td>
    <td>65</td>
    <td>3</td>
    <td>1</td>
    <td>31</td>
  </tr>
  <tr>
    <td>hawkes, self-correcting</td>
    <td>1</td>
    <td>64</td>
    <td>64</td>
    <td>train: 1406, testing: 156</td>
    <td>train: 1406, testing: 156</td>
    <td>train: 1406, testing: 156</td>
</table>

#### Generation of "data_hawkes", "data_hawkeshib", and "conttime":
The dataset 'data_hawkes' is generated by model of typical multivariate Hawkes process, 'data_hawkeshib' is generated by model of Hawkes Process with Inhibition (proposed by Hongyuan Mei), where positive constraints on base intensity and degree of activation are released, and 'conttime' is generated by the Neural Hawkes model. Events in each sequence are generated by Thinning algorithm.<br>
The main idea of Thinning algorithm for multivariate Hawkes Process is to generate time based on poisson process with intensity <img src="https://render.githubusercontent.com/render/math?math=\lambda^* \ge \lambda _k">, and accept the point with probability <img src="https://render.githubusercontent.com/render/math?math=\frac{\lambda_k}{\lambda^*}"> for each event type k in time (t_i, <img src="https://render.githubusercontent.com/render/math?math=\infty">) Then, we select the most recent time and type among K types for the next event (k<sub>i+1</sub>, t<sub>i+1</sub>).<br>

#### Description of MIMIC II Datasets
The Electron Medical Record (MIMIC II) is a collection of de-identified clinical visit of Intensice Care Unite patient for 7 years. Each event in the dataset is a record of its time stamp and disease diagnosis. 





## Acknowledgement:
This model is built by Hongrui Lyu, supervised by Hyunouk Ko and Dr. Huo. The file cont-time-cell is just a copy from Hongyuan Mei's code, but all other files are written by us. As notice by the original github page of pytorch implementation, this [license](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing/blob/master/LICENSE) need to be included. 

## Reference:

