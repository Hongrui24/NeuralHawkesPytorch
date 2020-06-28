import torch
import cont_time_cell
import torch.nn as nn


class Conttime(nn.Module):
    def __init__(self, n_types, beta = 0.1, hid_dim=32, lr=0.01):
        self.n_types = n_types
        self.beta = beta
        self.hid_dim = hid_dim

        super(Conttime, self).__init__()
        self.emb = nn.Embedding(self.n_types+1, self.hid_dim)
        self.lstm_cell = cont_time_cell.CTLSTMCell(hid_dim, beta)
        self.hidden_lambda = nn.Linear(self.hid_dim, self.n_types)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def train_batch(self, batch, sim_dur, total_time_lists, seq_len_lists, time_simulation_index):
        types, dtime = batch
        h_out, c_out, c_bar_out, decay_out, gate_out = self.forward(types, dtime)
        part_one_likelihood, part_two_likelihood, sum_likelihood = self.conttime_loss(h_out, c_out, c_bar_out,
                                                                                      decay_out, gate_out,
                                                                                      types, sim_dur, total_time_lists,
                                                                                      seq_len_lists, time_simulation_index)
        loss = -(torch.sum(part_one_likelihood - part_two_likelihood))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def forward(self, types, dtime):
        numb_seq, seq_len = dtime.shape
        self.hid_layer_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32)
        self.cell_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32)
        self.cell_bar_minus = torch.zeros(numb_seq, self.hid_dim, dtype=torch.float32)

        h_list, c_list, c_bar_list, decay_list, gate_out_list = [],[],[],[],[]
        for i in range(seq_len-1):
            type_input = self.emb(types[:,i])
            cell_i, cell_bar_updated, gate_decay, gate_output = self.lstm_cell(type_input, self.hid_layer_minus, self.cell_minus, self.cell_bar_minus)
            self.cell_minus, self.hid_layer_minus = self.lstm_cell.decay(cell_i, cell_bar_updated, gate_decay, gate_output, dtime[:,i+1])

            h_list.append(self.hid_layer_minus)
            c_list.append(cell_i)
            c_bar_list.append(cell_bar_updated)
            decay_list.append(gate_decay)
            gate_out_list.append(gate_output)
        h_out = torch.stack(h_list)
        c_out = torch.stack(c_list)
        c_bar_out = torch.stack(c_bar_list)
        decay_out = torch.stack(decay_list)
        gate_out = torch.stack(gate_out_list)
        # Length * Batch_size * hidden_dim
        return h_out, c_out, c_bar_out, decay_out, gate_out

    def conttime_loss(self, h, c, c_bar, decay, o, event_seqs, sim_duration, total_time_list, seq_len_lists,
                      time_simulation_index):
        # print(c)
        batch_size = event_seqs.shape[0]
        # print(time_simulation_index)
        sim_len = time_simulation_index.shape[1]
        part_one_likelihood = torch.zeros(batch_size)
        sum_likelihood = torch.zeros(batch_size)
        type_intensity = torch.nn.functional.softplus(self.hidden_lambda(h)).transpose(0,1)
        for idx in range(batch_size):
            event_seq = event_seqs[idx]
            seq_len = seq_len_lists[idx]
            part_one_likelihood[idx] = torch.sum(torch.log(type_intensity[idx, torch.arange(seq_len), event_seq[1:seq_len+1]]))
            sum_likelihood[idx] = torch.sum(torch.log(torch.sum(type_intensity[idx, torch.arange(seq_len), :], dim=-1)))

        c_sim = []
        c_bar_sim = []
        decay_sim = []
        o_sim = []
        for j in range(batch_size):
            layer_c = c[time_simulation_index[j], j, :]
            c_sim.append(layer_c)
            layer_c_bar = c_bar[time_simulation_index[j], j, :]
            c_bar_sim.append(layer_c_bar)
            layer_decay = decay[time_simulation_index[j], j, :]
            decay_sim.append(layer_decay)
            layer_o = o[time_simulation_index[j], j, :]
            o_sim.append(layer_o)
        c_sim = torch.stack(c_sim).transpose(0,1)
        c_bar_sim = torch.stack(c_bar_sim).transpose(0,1)
        decay_sim = torch.stack(decay_sim).transpose(0,1)
        o_sim = torch.stack(o_sim).transpose(0,1)
        h_sim_list = []
        # print(c_sim)
        for idx in range(sim_duration.shape[1]):
            cell_next, h_sim = self.lstm_cell.decay(c_sim[idx], c_bar_sim[idx], decay_sim[idx], o_sim[idx], sim_duration[:, idx])
            h_sim_list.append(h_sim)
        h_sim_list = torch.stack(h_sim_list)
        sim_intensity = torch.nn.functional.softplus(self.hidden_lambda(h_sim_list)).transpose(0,1)
        part_two_likelihood = torch.zeros(batch_size)
        for idx in range(batch_size):
            coefficient = total_time_list[idx] / sim_len
            part_two_likelihood[idx] = torch.sum(torch.sum(sim_intensity[idx, torch.arange(sim_len), :])) * coefficient


        return part_one_likelihood, part_two_likelihood, sum_likelihood





"""
The script here is a test to make sure the model will run.
The default type size is 4, and time is time duration.
Batch_size = 2, seq_len = 8+1
"""
if __name__ == "__main__":
    train_type = torch.tensor([[4,0,3,2,1,0,3,2,1],[4,0,3,2,1,0,3,0,0]])
    train_dtime = torch.tensor([[0, 0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2], [0, 0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0, 0]])
    sim_durations = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    sim_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 7], [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7]])
    seq_len_list = torch.tensor([8,6])
    total_time_list = torch.tensor([2.8, 2.8])
    # input to initialize the model include number of types
    model = Conttime(4)
    a_batch = (train_type, train_dtime)
    # The input to train include batch(padded time, padded type), padded_sim_durations, total_time_list, seq_len_list
    model.train_batch(a_batch, sim_durations, total_time_list, seq_len_list, sim_index)
