import torch
import pickle
from torch.utils.data import DataLoader

def open_pkl_file(path, description):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        data = data[description]
    time_durations = []
    type_seqs = []
    seq_lens = []
    for i in range(len(data)):
        seq_lens.append(len(data[i]))
        type_seqs.append(torch.LongTensor([int(event['type_event']) for event in data[i]]))
        time_durations.append(torch.FloatTensor([float(event['time_since_last_event']) for event in data[i]]))
    return time_durations, type_seqs, seq_lens



def open_txt_file(path):
    f = open(path, 'r')
    data_file = f.readlines()
    f.close()
    time_duration = []
    seq_lens_list = []
    # total_time_list = []
    for line in data_file:
        data = line.split(" ")
        a_list = []
        previous = 0
        lens = 0
        for i in range(len(data)):
            if data[i] != "\n":
                a_list.append(float(data[i]) - previous)
                previous = float(data[i])
                lens += 1
        time_duration.append(torch.tensor(a_list))
        # total_time_list.append(previous)
        seq_lens_list.append(lens)
    return time_duration, seq_lens_list


def get_index_txt(duration):
    type_list = []
    for i in range(len(duration)):
        a_list = torch.zeros(size=duration[i].shape, dtype=torch.long)
        type_list.append(a_list)
    type_list = torch.stack(type_list)
    return type_list


def padding_full(time_duration, type_train, seq_lens_list, type_size):
    max_len = max(seq_lens_list)
    batch_size = len(time_duration)
    time_duration_padded = torch.zeros(size=(batch_size, max_len+1))
    type_train_padded = torch.zeros(size=(batch_size, max_len+1), dtype=torch.long)
    for idx in range(batch_size):
        time_duration_padded[idx, 1:seq_lens_list[idx]+1] = time_duration[idx]
        type_train_padded[idx, 0] = type_size
        type_train_padded[idx, 1:seq_lens_list[idx]+1] = type_train[idx]
    return time_duration_padded, type_train_padded


def padding_seq_len(duration, types, type_size, seq_len):
    time_duration = []
    type_lists = []
    seq_lens_list = []
    batch_size = len(duration)
    for i in range(batch_size):
        end = seq_len
        while end <= duration[i].shape.__getitem__(-1):
            start = end - seq_len
            duration_list = [0]
            type_list = [type_size]
            duration_list = duration_list + duration[i][start:end].tolist()
            type_list = type_list + types[i][start:end].tolist()
            time_duration.append(duration_list)
            type_lists.append(type_list)
            seq_lens_list.append(seq_len)
            end += 1
    time_duration = torch.tensor(time_duration)
    type_lists = torch.tensor(type_lists)
    return time_duration, type_lists, seq_lens_list


def generate_simulation(durations, seq_len):
    max_seq_len = max(seq_len)
    simulated_len = max_seq_len * 5
    sim_durations = torch.zeros(durations.shape[0], simulated_len)
    sim_duration_index = torch.zeros(durations.shape[0], simulated_len, dtype=torch.long)
    total_time_seqs = []
    for idx in range(durations.shape[0]):
        time_seq = torch.stack([torch.sum(durations[idx][:i]) for i in range(1, seq_len[idx]+2)])
        total_time = time_seq[-1].item()
        total_time_seqs.append(total_time)
        sim_time_seq, _ = torch.sort(torch.empty(simulated_len).uniform_(0, total_time))
        sim_duration = torch.zeros(simulated_len)
        # print(sim_time_seq)
        # print(time_seq)

        for idx2 in range(time_seq.shape.__getitem__(-1)):
            duration_index = sim_time_seq > time_seq[idx2].item()
            sim_duration[duration_index] = sim_time_seq[duration_index] - time_seq[idx2]
            sim_duration_index[idx][duration_index] = idx2

        sim_durations[idx, :] = sim_duration[:]
    total_time_seqs = torch.tensor(total_time_seqs)
    # print(sim_duration_index)
    # print(total_time_seqs)
    # print(sim_durations)
    return sim_durations, total_time_seqs, sim_duration_index


class Data_Batch:
    def __init__(self, duration, events, seq_len):
        self.duration = duration
        self.events = events
        self.seq_len = seq_len

    def __len__(self):
        return self.events.shape[0]

    def __getitem__(self, index):
        sample = {
            'event_seq': self.events[index],
            'duration_seq': self.duration[index],
            'seq_len': self.seq_len[index]
        }
        return sample


if __name__ == "__main__":
    # time_duration, seq_lens_list = open_txt_file("data/hawkes/time_train.txt")
    # type_size = 1
    # type_train = get_index_txt(time_duration)
    # print(type_train)
    # time_duration_padded, type_train_padded = padding_full(time_duration, type_train, seq_lens_list, type_size)
    # time_duration_padded, type_train_padded, seq_len_list = padding_seq_len(time_duration, type_train, type_size, 20)
    # time_duration_padded = torch.tensor([[1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4]])
    # type_train_padded = torch.tensor([[11, 11, 11, 11, 11], [22, 22, 22, 22, 22], [33, 33, 33, 33, 33], [44, 44, 44, 44, 44]])
    # seq_len_list = torch.tensor([1,2,3,4])
    # train_data = Data_Batch(time_duration_padded, type_train_padded, seq_len_list)
    # train_data = DataLoader(train_data, batch_size=2, shuffle=True)
    # for i, train_batch in enumerate(train_data):
    #    print(train_batch)
    train_duration = torch.tensor([[0, 2, 1, 2, 3, 0, 1, 2, 3], [0, 2, 1, 2, 3, 0, 1, 0, 0]])
    seq_len = torch.tensor([8, 6])
    generate_simulation(train_duration, seq_len)
    # duration, types, seq_lens = open_pkl_file('data/conttime/train.pkl', 'train')
    # print(duration[0])
    # print(seq_lens[0])
    # time_duration, type_train = padding_full(duration, types, seq_lens, 8)
    # print(time_duration[0])
