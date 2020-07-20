import argparse
import utils
import sys
import os
import datetime
import time
from torch.utils.data import DataLoader
import conttime
import torch
import matplotlib.pyplot as plt
import test
import torch.nn.functional as F


def log_valid(time_duration, type_test, seq_lens, device):
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
    return log_likelihood, type_likelihood, time_likelihood

def type_valid(time_durations, seq_lens_lists, type_tests):
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
    
    total_numb = len(original_types)
    numb_correct = 0
    for idx in range(total_numb):
        if predicted_types[idx] == original_types[idx]:
            numb_correct += 1
    return numb_correct / total_numb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model..")

    parser.add_argument("--dataset", type=str, help="e.g. hawkes", required=True)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="maximum epochs")
    parser.add_argument("--seq_len", type=int, default=-1, help="truncated sequence length for hawkes and self-correcting, -1 means full sequence")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch_size for each train iteration")
    parser.add_argument("--used_past_model", type=bool, help="True to use a trained model named model.pt")

    config = parser.parse_args()

    dataset = config.dataset
    lr = config.lr
    seq_len = config.seq_len
    num_epochs = config.epochs
    batch_size = config.batch_size
    used_model = config.used_past_model

    now = str(datetime.datetime.today()).split()
    now = now[0]+"-"+now[1][:5]
    id_process = os.getpid()
    print("id: " + str(id_process))
    log_file_name = "train_process"+str(id_process)+".txt"
    log = open(log_file_name, 'w')
    log.write("Data when training: " + str(datetime.datetime.now()))
    log.write("\nTraining-id: " + str(id_process))
    log.write("\nTraining data: " + dataset)
    log.write("\nLearning rate: " + str(lr))
    log.write("\nMax epochs: " + str(num_epochs))
    log.write("\nseq lens: " + str(seq_len))
    log.write("\nbatch size for train: " + str(batch_size))
    log.write("\nuse previous model: " + str(used_model))

    t1 = time.time()
    print("Processing data...")
    if dataset == 'hawkes' or dataset == "self-correcting":
        file_path = 'data/' + dataset + "/time-train.txt"   # train file
        test_path = 'data/' + dataset + '/time-test.txt'    # test file
        time_duration, seq_lens_list = utils.open_txt_file(file_path)   # train time info
        test_duration, seq_lens_test = utils.open_txt_file(test_path)   # test time info
        type_size = 1
        type_train = utils.get_index_txt(time_duration) # train type
        type_test = utils.get_index_txt(test_duration)  # test type
        if seq_len == -1:
            time_duration, type_train = utils.padding_full(time_duration, type_train, seq_lens_list, type_size)
            test_duration, type_test = utils.padding_full(test_duration, type_test, seq_lens_test, type_size)
        else:
            time_duration, type_train, seq_lens_list = utils.padding_seq_len(time_duration, type_train, type_size, seq_len)
            test_duration, type_test = utils.padding_seq_len(test_duration, type_test, type_size, seq_len)
    else:
        if dataset == 'conttime' or dataset == "data_hawkes" or dataset == "data_hawkeshib":
            type_size = 8
        elif dataset == 'data_mimic1' or dataset == 'data_mimic2' or dataset == 'data_mimic3' or dataset == 'data_mimic4' or\
        dataset == 'data_mimic5':
            type_size = 75
        elif dataset == 'data_so1' or dataset == 'data_so2' or dataset == 'data_so3' or dataset == 'data_so4' or\
        dataset == 'data_so5':
            type_size = 22
        elif dataset == 'data_book1' or dataset == 'data_book2' or dataset == 'data_book3' or dataset == 'data_book4'\
        or dataset == 'data_book5':
            type_size = 3
        else:
            print("Data process file for other types of datasets have not been developed yet, or the dataset is not found")
            log.write("\nData process file for other types of datasets have not been developed yet, or the datase is not found")
            log.close()
            sys.exit()
        file_path = 'data/' + dataset + '/train.pkl'
        test_path = 'data/' + dataset + '/dev.pkl'
        time_duration, type_train, seq_lens_list = utils.open_pkl_file(file_path, 'train')
        test_duration, type_test, seq_lens_test = utils.open_pkl_file(test_path, 'dev')
        time_duration, type_train = utils.padding_full(time_duration, type_train, seq_lens_list, type_size)
        test_duration, type_test = utils.padding_full(test_duration, type_test, seq_lens_test, type_size)   

    train_data = utils.Data_Batch(time_duration, type_train, seq_lens_list)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print("Data Processing Finished...")
    t2 = time.time()
    data_process_time = t2 - t1
    print("Getting data takes: " + str(data_process_time) + " seconds")
    log.write("\n\nGetting data takes: " + str(data_process_time) + " seconds")

    print("start training...")
    t3 = time.time()
    if used_model:
        model = torch.load("model.pt")
    else:
        model = conttime.Conttime(n_types=type_size, lr=lr)

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

    loss_value = []

    log_test_list = []
    log_time_list = []
    log_type_list = []

    type_accuracy_list = []
    for i in range(num_epochs):
        loss_total = 0
        events_total = 0
        max_len = len(train_data)
        for idx, a_batch in enumerate(train_data):
            durations, type_items, seq_lens = a_batch['duration_seq'], a_batch['event_seq'], a_batch['seq_len']
            sim_durations, total_time_seqs, time_simulation_index = utils.generate_simulation(durations, seq_lens)
            type_items.to(device)
            durations.to(device)
            sim_durations.to(device)
            total_time_seqs.to(device)
            seq_lens.to(device)
            time_simulation_index.to(device)
            batch = (type_items, durations)
            loss = model.train_batch(batch, sim_durations, total_time_seqs, seq_lens, time_simulation_index)
            log_likelihood = -loss
            total_size = torch.sum(seq_lens)
            loss_total += log_likelihood.item()
            events_total += total_size.item()
            print("In epochs {0}, process {1} over {2} is done".format(i, idx, max_len))
        avg_log = loss_total / events_total
        loss_value.append(-avg_log)
        print("The log-likelihood at epochs {0} is {1}".format(i, avg_log))
        log.write("\nThe log likelihood at epochs {0} is {1}".format(i, avg_log))
        print("model saved..")
        torch.save(model, "model.pt")

        print("\nvalidating on log likelihood...")
        log_likelihood, type_likelihood, time_likelihood = log_valid(test_duration, type_test, seq_lens_test, device)
        log_test_list.append(-log_likelihood.item())
        log_type_list.append(-type_likelihood.item())
        log_time_list.append(-time_likelihood.item())

        print("\nvalidating on type prediction accuracy if we know when will next event happens...\n\n")
        accuracy = type_valid(test_duration, seq_lens_test, type_test)
        type_accuracy_list.append(accuracy)


    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    figure.suptitle(dataset + "'s Training Figure")
    ax[0].set_xlabel("epochs")
    ax[0].plot(loss_value, label='training loss')
    ax[0].plot(log_test_list, label='testing loss')
    ax[0].legend()
    ax[1].set_xlabel("epochs")
    ax[1].plot(log_type_list, label='testing type loss')
    ax[1].plot(log_time_list, label='testing time loss')
    ax[1].legend()
    ax[2].set_xlabel("epochs")
    ax[2].set_ylabel('accuracy')
    ax[2].set_title('type-validation-accuracy')
    ax[2].plot(type_accuracy_list, label='dev type accuracy')
    plt.subplots_adjust(top=0.85)
    figure.tight_layout()
    plt.savefig("training.jpg")

    t4 = time.time()
    training_time = t4 - t3
    print("training done..")
    print("training takes {0} seconds".format(training_time))
    log.write("\ntraining takes {0} seconds".format(training_time))
    log.close()

    print("Saving training loss and validation data...")
    print("If you have a trained model before this, please combine the previous train_date file to" +
        " generate plots that are able to show the whole training information")
    training_info_file = "training-data-" + now + ".txt"
    file = open(training_info_file, 'w')
    file.write("log-likelihood: ")
    file.writelines(str(item) + " " for item in loss_value)
    file.write('\nlog-test-likelihood: ')
    file.writelines(str(item) + " " for item in log_test_list)
    file.write('\nlog-type-likelihood: ')
    file.writelines(str(item) + " " for item in log_type_list)
    file.write('\nlog-time-likelihood: ')
    file.writelines(str(item) + " " for item in log_time_list)
    file.write('\naccuracy: ')
    file.writelines(str(item) + " " for item in type_accuracy_list)
    file.close()
    print("Every works are done!")




