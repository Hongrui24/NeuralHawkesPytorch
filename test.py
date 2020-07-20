import argparse
import utils
import sys
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


def test1(time_duration, seq_lens_list, type_test, n_sample, dataset):
    # read the model and simulate durations for integral
    print("testing...")
    print("generating test samples...")
    max_len = time_duration.shape[-1]
    estimated_times = []
    estimated_intensities = []
    estimated_types = []
    original_time = time_duration[0][57:].tolist()
    model = torch.load("model.pt")
    for idx in range(57, max_len):
        time_durations = time_duration[0][:idx]
        type_tests = type_test[0][:idx]
        max_duration = torch.max(time_durations)
        simulated_duration = torch.sort(torch.empty(n_samples).uniform_(0, 40*max_duration.item()))[0].reshape(n_samples, 1)
        time_durations = time_durations.expand(n_samples, time_durations.shape[-1])
        time_duration_sim_padded = torch.cat((time_durations, simulated_duration), dim=1)
        type_tests = type_tests.expand(n_samples, type_tests.shape[-1])

        # get the simulated out and processing h, c, c_bar, decay, o
        h_out, c_out, c_bar_out, decay_out, gate_out = model(type_tests, time_duration_sim_padded)
        simulated_h, simulated_c, simulated_c_bar, simulated_decay, simulated_o = h_out[-1], c_out[-1], c_bar_out[-1], decay_out[-1], gate_out[-1]
        h_last, c_last, c_bar_last, decay_last, o_last = h_out[-2][0], c_out[-2][0], c_bar_out[-2][0], decay_out[-2][0], gate_out[-2][0]

        # calculate estimated lambda and density
        # Use of monte carlo simulation of the way below:
        # p(di) = lambda(di) * exp(-(sum from 1 to i (lambda(dk))) * di / i) to calculate p(di) = lambda(di) * exp(-(integral from 0 to di (lambda(tao))))
        estimated_lambda_sum = torch.sum(F.softplus(model.hidden_lambda(simulated_h)), dim=-1)
        estimated_lambda_sum = estimated_lambda_sum.reshape(n_samples)
        simulated_duration = simulated_duration.reshape(n_samples)
        simulated_integral_exp_terms = torch.stack([(torch.sum(estimated_lambda_sum[:(i+1)]) * (simulated_duration[i] / (i+1))) for i in range(0, n_samples)])
        simulated_density = estimated_lambda_sum * torch.exp(-simulated_integral_exp_terms)
        estimated_time = torch.sum(simulated_duration * simulated_density) * (40 * max_duration.item()) / n_samples
        estimated_times.append(estimated_time.item())

        # calculate intensity and typies

        type_input = model.emb(type_tests[0][-1])
        cell_i, cell_bar_updated, gate_decay, gate_output = model.lstm_cell(type_input, h_last, c_last, c_bar_last)
        _, hidden = model.lstm_cell.decay(cell_i, cell_bar_updated, gate_decay, gate_output, estimated_time)
        lambda_all = F.softplus(model.hidden_lambda(hidden))
        _, estimated_type = torch.max(lambda_all, dim=-1)
        estimated_types.append(estimated_type.item())
        lambda_sum = torch.sum(lambda_all, dim=-1)
        estimated_intensities.append(lambda_sum.item())
        print("prediction at event {0} on a sequence of length {1} is done".format(idx, max_len))

    # print(len(estimated_times))
    # print(len(original_time))
    rmse = sqrt(mean_squared_error(original_time, estimated_times))
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    figure.suptitle(dataset+" by Neural Hawkes")
    ax[0].plot(original_time, label="actual")
    ax[0].plot(estimated_times, label="predicted")
    ax[0].set_xlabel("Time Index")
    ax[0].set_ylabel("Time Duration")
    ax[0].legend()
    ax[1].plot(estimated_intensities)
    ax[1].set_xlabel("Time Index")
    ax[1].set_ylabel("Intensity")
    ax[2].bar(x=1, height=rmse)
    ax[2].set_title("time RMSE")
    ax[2].annotate(str(round(rmse,3)),xy=[1, rmse])
    plt.subplots_adjust(top=0.45)
    figure.tight_layout()
    plt.savefig("result.png")
    print("testing done.. Please check the plots for estimated duration and intensity")
    return estimated_types


def test2(time_duration, type_test, seq_lens, device, log):
    print("start testing...")
    model = torch.load("model.pt")
    seq_lens = torch.tensor(seq_lens)
    sim_durations, total_time_seqs, time_simulation_index = utils.generate_simulation(time_duration, seq_lens)
    type_test.to(device)
    time_duration.to(device)
    sim_durations.to(device)
    total_time_seqs.to(device)
    seq_lens.to(device)
    time_simulation_index.to(device)
    h_out, c_out, c_bar_out, decay_out, gate_out = model(type_test, time_duration)
    part_one_likelihood, part_two_likelihood, sum_likelihood = model.conttime_loss(h_out, c_out, c_bar_out, decay_out, gate_out, type_test, sim_durations, total_time_seqs, seq_lens, time_simulation_index)
    total_size = torch.sum(seq_lens)
    log_likelihood = torch.sum(part_one_likelihood - part_two_likelihood) / total_size
    type_likelihood = torch.sum(part_one_likelihood - sum_likelihood) / total_size
    time_likelihood = log_likelihood - type_likelihood
    print("Test log likelihood is {0}".format(log_likelihood.item()))
    print("Test type log likelihood is {0}".format(type_likelihood.item()))
    print("Test time log likelihood is {0}".format(time_likelihood.item()))
    log.write("\nTest log likelihood is {0}".format(log_likelihood.item()))
    log.write("\nTest type log likelihood is {0}".format(type_likelihood.item()))
    log.write("\nTest time log likelihood is {0}".format(time_likelihood.item()))


def test3(time_durations, seq_lens_lists, type_tests, n_samples, dataset, log):
    model = torch.load("model.pt")
    numb_tests = time_durations.shape[0]
    original_types = []
    predicted_types = []
    for i in range(numb_tests):
        time_duration = time_durations[i:i+1]
        type_test = type_tests[i:i+1]
        seq_len = seq_lens_lists[i]

        original_types.append(type_test[0][seq_len].item())
        type_test = type_test[:, :seq_len]
        time_duration = time_duration[:, :seq_len+1]

        h_out, c_out, c_bar_out, decay_out, gate_out = model(type_test, time_duration)
        lambda_all = F.softplus(model.hidden_lambda(h_out[-1]))
        lambda_sum = torch.sum(lambda_all, dim=-1)
        lambda_all = lambda_all / lambda_sum
        # print(lambda_all)
        _, predict_type = torch.max(lambda_all, dim=-1)
        predicted_types.append(predict_type.item())
    log.write("\noriginal type:  ")
    for item in original_types:
        log.write(str(item))
        log.write(" ")
    log.write("\npredicted_type: ")
    for item in predicted_types:
        log.write(str(item))
        log.write(" ")
    
    total_numb = len(original_types)
    numb_correct = 0
    for idx in range(total_numb):
        if predicted_types[idx] == original_types[idx]:
            numb_correct += 1
    log.write("\ncorrectness: {0}".format(numb_correct/total_numb))

    error = 1 - numb_correct/total_numb
    figure, ax = plt.subplots(1,1,figsize=(3,3))
    figure.suptitle(dataset+" by Neural Hawkes")
    ax.bar(x=1, height=error)
    ax.set_title('type prediction error')
    ax.annotate(str(round(error,3)),xy=[1, error])
    plt.subplots_adjust(top=0.8)
    plt.savefig("result.png")

        

def test4(time_durations, seq_lens_lists, type_tests, n_samples, dataset, log):
    model = torch.load("model.pt")
    numb_tests = time_durations.shape[0]
    original_types = []
    predicted_types = []
    original_durations = []
    predicted_durations = []

    for i in range(numb_tests):
        time_duration = time_durations[i:i+1]
        type_test = type_tests[i:i+1]
        seq_len = seq_lens_lists[i]

        original_types.append(type_test[0][seq_len].item())
        type_test = type_test[:, :seq_len]
        original_durations.append(time_duration[0][seq_len].item())
        time_duration = time_duration[:, :seq_len]

        max_duration = torch.max(time_duration)
        simulated_duration = torch.sort(torch.empty(n_samples).uniform_(0, 40*max_duration.item()))[0].reshape(n_samples, 1)
        time_duration = time_duration.expand(n_samples, time_duration.shape[-1])
        time_duration_sim_padded = torch.cat((time_duration, simulated_duration), dim=1)
        type_test = type_test.expand(n_samples, type_test.shape[-1])

        # get the simulated out and processing h, c, c_bar, decay, o
        h_out, c_out, c_bar_out, decay_out, gate_out = model(type_test, time_duration_sim_padded)
        simulated_h, simulated_c, simulated_c_bar, simulated_decay, simulated_o = h_out[-1], c_out[-1], c_bar_out[-1], decay_out[-1], gate_out[-1]
        h_last, c_last, c_bar_last, decay_last, o_last = h_out[-2][0], c_out[-2][0], c_bar_out[-2][0], decay_out[-2][0], gate_out[-2][0]

        # calculate estimated lambda and density
        # Use of monte carlo simulation of the way below:
        # p(di) = lambda(di) * exp(-(sum from 1 to i (lambda(dk))) * di / i) to calculate p(di) = lambda(di) * exp(-(integral from 0 to di (lambda(tao))))
        estimated_lambda_sum = torch.sum(F.softplus(model.hidden_lambda(simulated_h)), dim=-1)
        estimated_lambda_sum = estimated_lambda_sum.reshape(n_samples)
        simulated_duration = simulated_duration.reshape(n_samples)
        simulated_integral_exp_terms = torch.stack([(torch.sum(estimated_lambda_sum[:(i+1)]) * (simulated_duration[i] / (i+1))) for i in range(0, n_samples)])
        simulated_density = estimated_lambda_sum * torch.exp(-simulated_integral_exp_terms)
        estimated_time = torch.sum(simulated_duration * simulated_density) * (40 * max_duration.item()) / n_samples
        predicted_durations.append(estimated_time.item())

        # calculate intensity and typies

        type_input = model.emb(type_test[0][-1])
        cell_i, cell_bar_updated, gate_decay, gate_output = model.lstm_cell(type_input, h_last, c_last, c_bar_last)
        _, hidden = model.lstm_cell.decay(cell_i, cell_bar_updated, gate_decay, gate_output, estimated_time)
        lambda_all = F.softplus(model.hidden_lambda(hidden))
        _, estimated_type = torch.max(lambda_all, dim=-1)
        predicted_types.append(estimated_type.item())
        print("process {0} over {1} is done".format(i, numb_tests))


    log.write("\noriginal type:  ")
    for item in original_types:
        log.write(str(item))
        log.write(" ")
    log.write("\npredicted_type: ")
    for item in predicted_types:
        log.write(str(item))
        log.write(" ")
    
    total_numb = len(original_types)
    numb_correct = 0
    for idx in range(total_numb):
        if predicted_types[idx] == original_types[idx]:
            numb_correct += 1
    log.write("\ncorrectness: {0}".format(numb_correct/total_numb))

    error = 1 - numb_correct/total_numb

    # print(len(estimated_times))
    # print(len(original_time))
    rmse = sqrt(mean_squared_error(original_durations, predicted_durations))
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    figure.suptitle(dataset+" by Neural Hawkes")
    ax[0].plot(original_durations, label="actual")
    ax[0].plot(predicted_durations, label="predicted")
    ax[0].set_xlabel("Sequence Index")
    ax[0].set_ylabel("Time Duration")
    ax[0].legend()
    ax[1].bar(x=1, height=rmse)
    ax[1].set_ylabel("time RMSE")
    ax[1].annotate(str(round(rmse,3)),xy=[1, rmse])
    ax[2].bar(x=1, height=error)
    ax[2].set_ylabel("Type Prediction Error")
    ax[2].annotate(str(round(error,3)),xy=[1, error])
    plt.subplots_adjust(top=0.5)
    figure.tight_layout()
    plt.savefig("result.png")
    print("testing done.. Please check the plots for estimated duration and type intensity")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="e.g hawkes. Must be the same as the training data in training model")
    parser.add_argument("--test_type", type=int,
                        help="What test to do. 1 - duration, intensity, and rmse plot in KDD using KDD's data;2 - log likelihood test" + 
                        "3 - test on accuracy of types of known duration;4 - test on accuracy of duration and types of unknown duration",
                        default=1)
    parser.add_argument("--n_samples", type=int, help="number of samples of monte carlo integration in test 1",
                        default=10000)

    id_process = os.getpid()
    config = parser.parse_args()
    dataset = config.dataset
    test_type = config.test_type
    n_samples = config.n_samples
    print("Testing...")
    log_file_name = "test_process"+str(id_process)+".txt"
    log = open(log_file_name, 'w')

    log.write("\nTest dataset: " + dataset)
    log.write("\nTest type: " + str(test_type))

    print("processing testing data set")
    if dataset == 'hawkes' or dataset == "self-correcting":
        file_path = 'data/' + dataset + "/time-test.txt"
        time_duration, seq_lens_list = utils.open_txt_file(file_path)
        type_size = 1
        type_test = utils.get_index_txt(time_duration)
        time_duration, type_test = utils.padding_full(time_duration, type_test, seq_lens_list, type_size)
    elif dataset == 'conttime' or dataset == "data_hawkes" or dataset == "data_hawkeshib": 
        file_path = 'data/' + dataset + "/test.pkl"
        type_size = 8
        time_duration, type_test, seq_lens_list = utils.open_pkl_file(file_path, 'test')
        time_duration, type_test = utils.padding_full(time_duration, type_test, seq_lens_list, type_size)
    elif dataset == 'data_mimic1' or dataset == 'data_mimic2' or dataset == 'data_mimic3' or dataset == 'data_mimic4' or dataset == 'data_mimic5':
        file_path = 'data/' + dataset + "/test.pkl"
        type_size = 75
        time_duration, type_test, seq_lens_list = utils.open_pkl_file(file_path, 'test')
        time_duration, type_test = utils.padding_full(time_duration, type_test, seq_lens_list, type_size)
    elif dataset == 'data_so1' or dataset == 'data_so2' or dataset == 'data_so3' or dataset == 'data_so4' or dataset == 'data_so5':
        file_path = 'data/' + dataset + "/test.pkl"
        type_size = 22
        time_duration, type_test, seq_lens_list = utils.open_pkl_file(file_path, 'test')
        time_duration, type_test = utils.padding_full(time_duration, type_test, seq_lens_list, type_size)
    elif dataset == 'data_book1' or dataset == 'data_book2' or dataset == 'data_book3' or dataset == 'data_book4'\
    or dataset == 'data_book5':
        file_path = 'data/' + dataset + "/test.pkl"
        type_size = 3
        time_duration, type_test, seq_lens_list = utils.open_pkl_file(file_path, 'test')
        time_duration, type_test = utils.padding_full(time_duration, type_test, seq_lens_list, type_size)
    else:
        print("Data process file for other types of datasets have not been developed yet, or the dataset is not found")
        log.close()
        sys.exit()

    print("testing dataset is done")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        log.write("\nYou are using GPU acceleration.")
        print("Number of GPU: ", torch.get_num_threads())
        log.write("\n\nNumber of GPU: " + str((torch.get_num_threads())))
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        log.write("\nCUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())
        log.write("\n\nNumber of cores: " + str(os.cpu_count()))

    if test_type == 1:
        time_duration = time_duration[:1]
        seq_lens_list = seq_lens_list[:1]
        type_test = type_test[:1]
        estimated_types = test1(time_duration, seq_lens_list, type_test, n_samples, dataset)
        log.write("\nestimated types: \n")
        for item in estimated_types:
            log.write(str(item))
            log.write(" ")
    elif test_type == 2:
        test2(time_duration, type_test, seq_lens_list, device, log)
    elif test_type == 3:
        test3(time_duration, seq_lens_list, type_test, n_samples, dataset, log)
    elif test_type == 4:
        test4(time_duration, seq_lens_list, type_test, n_samples, dataset, log)
    else:
        print("Other tests have not been developed yet.")
        log.close()
        sys.exit()
    log.close()



