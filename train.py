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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model..")

    parser.add_argument("--dataset", type=str, help="e.g. hawkes", required=True)
    parser.add_argument("--lr", type=int, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="maximum epochs")
    parser.add_argument("--seq_len", type=int, default=-1, help="truncated sequence length, -1 means full sequence")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch_size for each train iteration")
    parser.add_argument("--used_past_model", type=bool, help="True to use a trained model named model.pt")

    config = parser.parse_args()

    dataset = config.dataset
    lr = config.lr
    seq_len = config.seq_len
    num_epochs = config.epochs
    batch_size = config.batch_size
    used_model = config.used_past_model

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
    log.write("\nuse previous model:" + str(used_model))

    t1 = time.time()
    print("Processing data...")
    if dataset == 'hawkes':
        file_path = 'data/' + dataset + "/time_train.txt"
        time_duration, seq_lens_list = utils.open_txt_file(file_path)
        type_size = 1
        type_train = utils.get_index_txt(time_duration)
        if seq_len == -1:
            time_duration, type_train = utils.padding_full(time_duration, type_train, seq_lens_list, type_size)
        else:
            time_duration, type_train, seq_lens_list = utils.padding_seq_len(time_duration, type_train, type_size, seq_len)
    elif dataset == 'conttime' or dataset == "data_hawkes" or dataset == "data_hawkeshib":
        file_path = 'data/' + dataset + '/train.pkl'
        type_size = 8
        time_duration, type_train, seq_lens_list = utils.open_pkl_file(file_path, 'train')
        time_duration, type_train = utils.padding_full(time_duration, type_train, seq_lens_list, type_size)
    else:
        print("Data process file for other types of datasets have not been developed yet, or the dataset is not found")
        log.write("\nData process file for other types of datasets have not been developed yet, or the datase is not found")
        log.close()
        sys.exit()

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
        model = conttime.Conttime(n_types=type_size+1, lr=lr)

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
        loss_value.append(avg_log)
        print("The loss at epochs {0} is {1}".format(i, avg_log))
        log.write("\nThe loss at epochs {0} is {1}".format(i, avg_log))
        print("model saved..")
        torch.save(model, "model.pt")
    plt.plot(loss_value)
    plt.savefig("log-loss-graph.jpg")
    t4 = time.time()
    training_time = t4 - t3
    print("training done..")
    print("training takes {0} seconds".format(training_time))
    log.write("\ntraining takes {0} seconds".format(training_time))
    log.close()



