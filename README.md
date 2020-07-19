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
![training2](https://user-images.githubusercontent.com/54515153/87878384-e6651500-c9b1-11ea-9976-bf79efe98213.JPG)
![training3](https://user-images.githubusercontent.com/54515153/87862406-bec46d00-c91d-11ea-9246-a1a0edef354e.JPG)
![training4](https://user-images.githubusercontent.com/54515153/87862407-c08e3080-c91d-11ea-83cb-c99e1cbc090e.JPG)
![training5](https://user-images.githubusercontent.com/54515153/87862408-c1bf5d80-c91d-11ea-9a79-61e9eb1567f3.JPG)

#### Prediction on Time (duration) and types:
![testing1](https://user-images.githubusercontent.com/54515153/87878387-e8c76f00-c9b1-11ea-87bd-98409d5f44cd.JPG)
![testing2](https://user-images.githubusercontent.com/54515153/87878388-ea913280-c9b1-11ea-8d94-bd819ed4b949.JPG)
![testing3](https://user-images.githubusercontent.com/54515153/87878389-ec5af600-c9b1-11ea-915d-de6e0fce9e1e.JPG)


## Running the Code:
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


## Testing:
### Data Source and Structure:
We use the data provided by the Hongyuan Mei and Du Nan to do tests. 
<table>
  <tr>
    <th>Name</th>
    <th>Type of Dataset</th>
    <th>Number of types</th>
    <th>Number of training sequence</th>
    <th>Number of testing sequence</th>
    <th>Sequence Length Mean</th>
    <th>Sequence Length Min</th>
    <th>Sequence Length Max</th>
  </tr>
  <tr>
    <td>data_hawkes, data_hawkeshib, data_conttime</td>
    <td>Simulated</td>
    <td>5</td>
    <td>8000</td>
    <td>1000</td>
    <td>60</td>
    <td>20</td>
    <td>100</td>
  </tr>
  <tr>
    <td>MIMIC-II(1)(2)(3)(4)(5)</td>
    <td>Real World Dataset</td>
    <td>75</td>
    <td>527</td>
    <td>65</td>
    <td>3</td>
    <td>1</td>
    <td>31</td>
  </tr>
  <tr>
    <td>SO(Stack Overflow) (1)(2)(3)(4)(5)</td>
    <td>Real World Dataset</td>
    <td>22</td>
    <td>4777</td>
    <td>1326</td>
    <td>72</td>
    <td>41</td>
    <td>736</td>
  <tr>
    <td>hawkes, self-correcting</td>
    <td>Simulated</td>
    <td>1</td>
    <td>64</td>
    <td>64</td>
    <td>train: 1406, testing: 156</td>
    <td>train: 1406, testing: 156</td>
    <td>train: 1406, testing: 156</td>
</table>
Description of MIMIC II Datasets<br>
The Electron Medical Record (MIMIC II) is a collection of de-identified clinical visit of Intensice Care Unite patient for 7 years. Each event in the dataset is a record of its time stamp and disease diagnosis. <br>
Description of SO (Stack Overflow) Datasets<br>
The Stack Overflow dataset represents two years of user awards on a question-answering website: each user received a sequence of badges<br>

Notice:<br>

The dataset 'data_hawkes', 'data_hawkeshib', 'conttime', 'hawkes', 'data_so', and 'self-correcting' in this repository is truncated from the original data due to uploading difficulty and long training time. You may train the data in this repository with more epochs to get a similar result below. The original dataset can be found in [this page](https://github.com/dunan/NeuralPointProcess/tree/master/data/synthetic) and [here](https://github.com/HMEIatJHU/neurawkes/tree/master/data)

<br><br><br>
### Test Results

#### Log-Likelihood Test on 'data_conttime'

The first test we do is to calculate average log-likelihood of events in test file of "data_conttime", and compare the results in Hongyuan Mei's paper. The model is trained with lr = 0.01, epochs = 30, mini batch size = 10.<bre />
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
<br>
We use this test to verify that our pytorch implementation of Neural Hawkes is the Neural Hawkes model described in Hongyuan Mei's paper. 
<br/><br/><br>


#### Test on 'hawkes' and 'self-correcting':
2. We also test out model with data provided in Du, Nan, et al. [“Recurrent Marked Temporal Point Processes.”](https://www.kdd.org/kdd2016/subtopic/view/recurrent-temporal-point-process) paper about self-correcting and hawkes. We make predictions on inter-event durations, intensities, and calculate RMSE between real inter-event durations and our predictions for events in a test sequence. We also compare the results with Du Nan's RMTPP's prediction and optimal prediction. We train the model for 10 epochs with learning rate = 0.01 and truncated sequence length = 75. <br />
- Result of "hawkes":
![result](https://user-images.githubusercontent.com/54515153/87882323-5765f600-c9cd-11ea-887e-9ee920b41900.png)
![Hawkes](https://user-images.githubusercontent.com/54515153/87882417-dd823c80-c9cd-11ea-846e-fe50ad0a0204.JPG)

- Result of "self-correcting"
![result](https://user-images.githubusercontent.com/54515153/87882330-5f259a80-c9cd-11ea-9d32-d19841026baf.png)
![Self-correcting](https://user-images.githubusercontent.com/54515153/87882419-e07d2d00-c9cd-11ea-9b3e-12004245fb7c.JPG)
 <br>
 This test show that Neural Hawkes model has the ability to achieve the prediction by optimal equation (prediction made by actual equation behind the dataset) for hawkes and self-correcting.
 <br><br><br>
 
 #### Test on 'MIMIC-II' and 'SO'
The third test we do is to test on type prediction accuracy. We choose two dataset to do the test: 'MIMICii' and 'SO' (Stack Overflow). For testing, we input a sequence in testing file except the last event to the trained model trained by 'train.pkl' and compare the model prediction with the actual one for the last event. For testing purpose, we also look at how loss and prediction accuracy on types changes with number of epochs, and we compare the type prediction accuracy with the prediction accuracy by pytorch implementation of RMTPP.<br><br>
Model During Training:
![training](https://user-images.githubusercontent.com/54515153/87884965-14614e00-c9e0-11ea-821b-ff3182ae2d01.jpg)
<br>
Testing Results on MIMIC-II:
<table>
  <tr>
    <th>Dataset</th>
    <th>(# epochs, lr)Error by Neural Hawkes</th>
    <th>(# epochs, lr)Error by RMTPP</th>
  </tr>
  <tr>
    <td>data_mimic1</td>
    <td>(200, 0.001) 10.8%</td>
    <td>(700, 0.0005) 20%</td>
  </tr>
  <tr>
    <td>data_mimic2</td>
    <td>(300, 0.001) 16.9%</td>
    <td>(900, 0.0005) 38.5%</td>
  </tr>
  <tr>
    <td>data_mimic3</td>
    <td>(200, 0.001) 16.9%</td>
    <td>(900, 0.0005) 32.3%</td>
  </tr>
  <tr>
    <td>data_mimic4</td>
    <td>(200, 0.001) 20% </td>
    <td>(2000, 0.0002)36.9% </td>
  </tr>
  <tr>
    <td>data_mimic5</td>
    <td>(200, 0.001) 9.2%</td>
    <td>(2000, 0.0002) 35.4%</td>
  </tr>
  <tr>
    <td>MIMIC-II Average</td>
    <td>(---, ----)14.76%</td>
    <td>(---, ----)32.62%</td>
  </tr>
</table>
<br><br>
Testing Results on SO dataset:
<table>
  <tr>
    <th>Dataset</th>
    <th> (epochs, lr)Error by Neural Hawkes</th>
  </tr>
  <tr>
    <td>data_so1</td>
    <td>(20, 0.01) 62%</td>
  </tr>
  <tr>
    <td>data_so1</td>
    <td>(20, 0.01) 61.5</td>
  </tr>
  <tr>
    <td>data_so1</td>
    <td>(20, 0.01) 59.5%</td>
  </tr>
  <tr>
    <td>data_so1</td>
    <td>(20, 0.01) 63%</td>
  </tr>
  <tr>
    <td>data_so1</td>
    <td>(20, 0.01) 62.3%</td>
  </tr>
  <tr>
    <td>average</td>
    <td>(--, ---) 61.66%</td>
  </tr>
</table>
<br>

#### Thoughts on Testing Results:
The prediction on types achieve a lower error rate on MIMIC-II dataset than Stack Overflow dataset. This may caused by a simplier type sequence on MIMIC-II dataset. That is event types in single sequence of MIMIC-II seldom changes. The following is a sample print out of a sequence in MIMIC-II and SO:<br>
Sample MIMIC-II Sequence:<br>
![mimic](https://user-images.githubusercontent.com/54515153/87886526-eb46ba80-c9eb-11ea-9f00-e88ec72f93d3.JPG)<br><br>
Sample SO Sequence:<br>
![so](https://user-images.githubusercontent.com/54515153/87886575-5c866d80-c9ec-11ea-80fc-63c911c810a2.JPG)<br>
Thus, the better prediction on types in MIMIC-II dataset may caused by the the recurrence of same event type in each sequence. 

## Acknowledgement:
This model is built by Hongrui Lyu, supervised by Hyunouk Ko and Dr. Huo. The file cont-time-cell is just a copy from Hongyuan Mei's code, but all other files are written by us. As notice by the original github page of pytorch implementation, this [license](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing/blob/master/LICENSE) need to be included. 


