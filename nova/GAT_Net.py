import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

# Input all obs, get all prob dist
class GAT_Net(nn.Module):
    def __init__(self, input_size, max_vehicle_num, GAT_hidden_dim, attention_dim, use_cuda=True):
        super(GAT_Net, self).__init__()
        # Decoding input own h_i and x_i, output the prob dist
        self.input_size = input_size
        self.max_vehicle_num = max_vehicle_num
        self.rnn_hidden_dim = GAT_hidden_dim
        self.attention_dim = attention_dim
        self.use_cuda = use_cuda

        # Encode each observation
        self.encoding = nn.Linear(input_size, self.rnn_hidden_dim)

        # Hard
        # GRU input: [[h_i,h_1],[h_i,h_2],...[h_i,h_n]] and [0,...,0]
        #     Output[[h_1],[h_2],...,[h_n]] and [h_n],
        # h_j represents the relation between agent j and agent i
        # Input dim = (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)，
        # i.e. For batch_size data, feed each agent's connection with other n_agents - 1 agents' hidden_state
        self.hard_bi_GRU = nn.GRU(self.rnn_hidden_dim * 2, self.rnn_hidden_dim, bidirectional=True)
        # Analyze h_j,get agent j's weight wrt agent i, output dim = 2, grab one of them after gumble_softmax
        # If 1, then consider agent_j
        # Bi-direction GRU, hidden_state dim = 2 * hidden_dim
        self.hard_encoding = nn.Linear(self.rnn_hidden_dim * 2, 2)

        # Soft
        self.q = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.k = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.rnn_hidden_dim, self.attention_dim)

        # Get hidden state from self obs
        # Each agent gets hidden_state from own obs to store prev obs
        self.rnn = nn.GRUCell(self.attention_dim, self.attention_dim)

    def forward(self, obs, hidden_state):
        # For agent_i, choose history_i [batch_size, max_vehicle_num, obs_dim]
        # batch_size = (1 (select action) / max_episode_len) * n_thread
        # batch_size * n_agents
        n_thread, max_vehicle_num, obs_dim = obs.shape
        size = n_thread * max_vehicle_num

        # Encode the obs
        obs_encoding = f.relu(self.encoding(obs))
        # Get h from own GRU, dim = (batch_size * max_vehicle_num, args.rnn_hidden_dim)
        h_out = obs_encoding.reshape(-1, self.rnn_hidden_dim)

        # Hard Attention, GRU input dim = (seq_len, batch_size, dim)
        # Reshape h to have n_agents dim, (batch_size, max_vehicle_num, rnn_hidden_dim)
        h = h_out.reshape(-1, self.max_vehicle_num, self.rnn_hidden_dim)

        input_hard = []
        for i in range(self.max_vehicle_num):
            # (batch_size, rnn_hidden_dim)
            h_i = h[:, i]
            h_hard_i = []
            # For agent i, concatenate h_i wth other agent's h
            for j in range(self.max_vehicle_num):
                if j != i:
                    h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
            # After the loop, h_hard_i is a list with n_agents - 1 tensor
            # with dim = (batch_size, rnn_hidden_dim * 2)
            h_hard_i = torch.stack(h_hard_i, dim=0)
            input_hard.append(h_hard_i)

        # After loop, input_hard is a list with n_agents tensor
        # with dim=(max_vehicle_num - 1, batch_size, max_vehicle_num, rnn_hidden_dim * 2)
        input_hard = torch.stack(input_hard, dim=-2)
        # Finally get (max_vehicle_num - 1, batch_size * max_vehicle_num, rnn_hidden_dim * 2) for input
        input_hard = input_hard.view(self.max_vehicle_num - 1, -1, self.rnn_hidden_dim * 2)

        # Bi-direction GRU, each GRU with 1 layer, so 1st layer is 2 * 1
        h_hard = torch.zeros((2 * 1, size, self.rnn_hidden_dim))
        if self.use_cuda:
            h_hard = h_hard.cuda()

        # (max_vehicle_num - 1,batch_size * max_vehicle_num,rnn_hidden_dim * 2)
        h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)
        # (batch_size * max_vehicle_num, max_vehicle_num - 1, rnn_hidden_dim * 2)
        h_hard = h_hard.permute(1, 0, 2)
        # (batch_size * max_vehicle_num * (max_vehicle_num - 1), rnn_hidden_dim * 2)
        h_hard = h_hard.reshape(-1, self.rnn_hidden_dim * 2)

        # Get hard weight, (max_vehicle_num, batch_size, 1,  max_vehicle_num - 1) with an extra dim for sum
        # (batch_size * max_vehicle_num * (max_vehicle_num - 1), 2)
        hard_weights = self.hard_encoding(h_hard)
        # Determine send to agent j or not (one-hot vector)
        hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)

        hard_weights = hard_weights[:, 1].view(-1, self.max_vehicle_num, 1, self.max_vehicle_num - 1)
        # (max_vehicle_num, batch_size, 1, (max_vehicle_num - 1))
        hard_weights = hard_weights.permute(1, 0, 2, 3)

        # Soft Attention
        # (batch_size, max_vehicle_num, args.attention_dim)
        q = self.q(h_out).reshape(-1, self.max_vehicle_num, self.attention_dim)
        # (batch_size, n_agents, args.attention_dim)
        k = self.k(h_out).reshape(-1, self.max_vehicle_num, self.attention_dim)
        # (batch_size, n_agents, args.attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.max_vehicle_num, self.attention_dim)

        x = []
        for i in range(self.max_vehicle_num):
            # agent i's q, (batch_size, 1, args.attention_dim)
            q_i = q[:, i].view(-1, 1, self.attention_dim)
            # Other agent's k and v
            k_i = [k[:, j] for j in range(self.max_vehicle_num) if j != i]
            v_i = [v[:, j] for j in range(self.max_vehicle_num) if j != i]

            # (max_vehicle_num - 1, batch_size, args.attention_dim)
            k_i = torch.stack(k_i, dim=0)
            # Exchange dimensions into (batch_size, args.attention_dim, max_vehicle_num - 1)
            k_i = k_i.permute(1, 2, 0)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            # (batch_size, 1, attention_dim) * (batch_size, attention_dim, max_vehicle_num - 1) = (batch_size, 1, max_vehicle_num - 1)
            score = torch.matmul(q_i, k_i)

            # Normalize
            scaled_score = score / np.sqrt(self.attention_dim)

            # softmax to get the weight, dim = (batch_size, 1, max_vehicle_num - 1)
            soft_weight = f.softmax(scaled_score, dim=-1)

            # Weighted sum get (batch_size, args.attention_dim)
            x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
            x.append(x_i)

        # Concatenate each agent's h and x
        # dim = (batch_size * max_vehicle_num, args.attention_dim)
        x = torch.stack(x, dim=1).reshape(-1, self.attention_dim)

        # # Read the hidden state and the attention state to generate the action
        h_out = self.rnn(x, hidden_state)
        # hidden dim = (batch_size * max_vehicle_num, args.attention_dim)
        return h_out

class MotionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, attention_dim):
        super(MotionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim

        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, attention_dim)

    def forward(self, input, hidden):
        input = f.relu(self.linear(input))
        input = input.view(-1, self.hidden_size)
        h_out = self.gru(input, hidden)
        return h_out  # (batch_size * max_vehicle_num, attention_dim)

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class AttenPoolNet(nn.Module):
    def __init__(self, hidden_size, mlp_dim, bottleneck_dim,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(AttenPoolNet, self).__init__()
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.bottleneck_dim = bottleneck_dim

        mlp_pre_dim = 2 * hidden_size
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, hidden_size)
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims,
                                    activation=activation,
                                    batch_norm=batch_norm,
                                    dropout=dropout)
        # Additional layers for processing velocity and computing attention weights
        self.velocity_embedding = nn.Linear(2, hidden_size)
        self.attention_mlp = make_mlp([hidden_size * 2, mlp_dim, 1],
                                      activation=activation, batch_norm=batch_norm, dropout=dropout)

    def compute_attention_weights(self, pos_embedding, velocity_embedding):
        concatenated = torch.cat([pos_embedding, velocity_embedding], dim=1)
        attention_scores = self.attention_mlp(concatenated)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights

    def forward(self, hidden, additional):
        position = additional[..., :2].view(-1, 2)  # (batch_size * max_vehicle_num, 2)
        velocity = additional[..., 2:].view(-1, 2)  # (batch_size * max_vehicle_num, 2)

        pos_embedding = self.spatial_embedding(position)  # (batch_size * max_vehicle_num, hidden_size)
        velocity_embedding = self.velocity_embedding(velocity)  # (batch_size * max_vehicle_num, hidden_size)

        attention_weights = self.compute_attention_weights(pos_embedding, velocity_embedding)
        weighted_h_input = torch.cat([pos_embedding, hidden], dim=1)  # (batch_size * max_vehicle_num, hidden_size + attention_size)
        weighted_h_input *= 0.05 * attention_weights.view(-1, 1)

        # MLP processing as in PoolHiddenNet
        pooling_h = self.mlp_pre_pool(weighted_h_input)  # (batch_size * max_vehicle_num, bottleneck_dim)
        
        return pooling_h


class PredictionEncoder(nn.Module):
    def __init__(self, input_size, args):
        super(PredictionEncoder, self).__init__()
        self.input_size = input_size
        self.args = args
        self.max_vehicle_num = args.max_vehicle_num
        self.hidden_size = args.GAT_hidden_dim
        self.attention_dim = args.attention_dim
        self.latent_dim = args.latent_dim
        self.use_cuda = args.use_cuda
        self.mlp_dim = args.mlp_dim
        self.dropout = args.dropout
        self.bottleneck_dim = args.bottleneck_dim

        self.gat_net = GAT_Net(input_size=input_size - 5,
                               max_vehicle_num=self.max_vehicle_num,
                               GAT_hidden_dim=self.hidden_size,
                               attention_dim=self.attention_dim,
                               use_cuda = self.use_cuda)
        self.motion_encoder = MotionEncoder(input_size=4,
                                            hidden_size=self.hidden_size,
                                            attention_dim=self.attention_dim)
        self.mlp = make_mlp([self.attention_dim * 2, self.mlp_dim, self.attention_dim],
                            activation='relu', batch_norm=True, dropout=self.dropout)
        self.attn_pool_net = AttenPoolNet(hidden_size=self.hidden_size,
                                          mlp_dim=self.mlp_dim,
                                          bottleneck_dim=self.bottleneck_dim,
                                          dropout=self.dropout)

    def forward(self, input, hidden):
        # input: (pred_batch_size, max_vehicle_num, obs_dim + latent_dim)
        # hidden: (pred_batch_size * max_vehicle_num, attention_dim * 2)
        obs = input[..., : -self.latent_dim]
        behavior_latent = input[..., -self.latent_dim:]
        input_gat = torch.cat([obs[..., :-5], behavior_latent], dim=-1)
        input_motion = obs[..., -5: -1]
        input_add = obs[..., 1: 5]  # additional inputs: (x, y, vx, vy)

        hidden_gat  = hidden[:, :self.attention_dim].contiguous()
        hidden_motion = hidden[:, self.attention_dim:].contiguous()

        out_gat = self.gat_net(input_gat, hidden_gat)
        out_motion = self.motion_encoder(input_motion, hidden_motion)

        hidden_combined = torch.cat([out_gat, out_motion], dim=-1)  # (pred_batch_size * max_vehicle_num, attention_dim * 2)
        hidden_processed = self.mlp(hidden_combined)  # (pred_batch_size * max_vehicle_num, attention_dim)
        out_pool = self.attn_pool_net(hidden_processed, input_add)  # (pred_batch_size * max_vehicle_num, bottleneck_dim)

        return out_pool, hidden_combined