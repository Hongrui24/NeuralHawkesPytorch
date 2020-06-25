import argparse
import utils
import sys
import torch
import os


def test1():
    pass


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="e.g hawkes. Must be the same as the training data in training model")
    parser.add_argument("--test_type", type=int,
                        help="What test to do. 1 - duration, intensity, and rmse plot in KDD using KDD's data;-2 log likelihood test",
                        default=1)
    parser.add_argument("--n_samples", type=int, help="number of samples in monte carlo integration",
                        default=5000)

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
    if dataset == 'hawkes':
        file_path = 'data/' + dataset + "/time_test.txt"
        time_duration, seq_lens_list = utils.open_txt_file(file_path)
        type_size = 1
        type_test = utils.get_index_txt(time_duration)
        time_duration = torch.stack(time_duration)
        seq_lens_list = torch.tensor(seq_lens_list)
    elif dataset == 'conttime' or dataset == "data_hawkes" or dataset == "data_hawkeshib":
        file_path = 'data/' + dataset + "/test.pkl"
        type_size = 8
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
        test1()
    elif test_type == 2:
        test2(time_duration, type_test, seq_lens_list, device, log)
    else:
        print("Other tests have not been developed yet.")
        log.close()
        sys.exit()
    log.close()



