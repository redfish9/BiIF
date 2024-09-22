import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT_Net(nn.Module):
    def __init__(self, input_size, GAT_hidden_size):
        super(GAT_Net, self).__init__()
        self.input_size = input_size
        self.GAT_hidden_size = GAT_hidden_size

        self.W = nn.Linear(input_size, GAT_hidden_size, bias=False)
        self.a = nn.Linear(GAT_hidden_size * 2, 1, bias=False)

    def forward(self, features):
        # features:
        # - use_behavior: (batch_size, max_vehicle_num, hidden_size*2+latent_dim)
        # - else: (batch_size, max_vehicle_num, hidden_size*2)
        N, V, _ = features.size()
        h = self.W(features)
        
        a_input = torch.cat([h.repeat(1, 1, V).view(N, V * V, -1), h.repeat(1, V, 1)], dim=2).view(N, V, V, 2 * self.GAT_hidden_size)
        e = F.leaky_relu(self.a(a_input).squeeze(3))
        
        attention = F.softmax(e, dim=2)
        h_prime = torch.bmm(attention, h)
        
        return h_prime

class PredictionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, GAT_use_behavior, latent_dim,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(PredictionEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.GAT_use_behavior = GAT_use_behavior
        self.latent_dim = latent_dim

        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU()
        )
        self.velocity_embedding = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU()
        )
        if self.GAT_use_behavior:
            self.gat = GAT_Net(input_size=hidden_size*2+latent_dim, GAT_hidden_size=hidden_size)
        else:
            self.gat = GAT_Net(input_size=hidden_size*2, GAT_hidden_size=hidden_size)
        self.mlp = make_mlp([hidden_size*2, hidden_size, output_size],
                            activation=activation, 
                            batch_norm=batch_norm, 
                            dropout=dropout)

    def forward(self, obs, high_intention):
        spatial_embedding = self.spatial_embedding(obs[:, :, 1:3])
        velocity_embedding = self.velocity_embedding(obs[:, :, 3:5])
        if self.GAT_use_behavior:
            combined_embedding = torch.cat((spatial_embedding, velocity_embedding, high_intention), dim=-1)
        else:
            combined_embedding = torch.cat((spatial_embedding, velocity_embedding), dim=-1)
        gat_output = self.gat(combined_embedding)
        concat_features = torch.cat((gat_output, spatial_embedding), dim=-1)
        mlp_out = self.mlp(concat_features.view(-1, self.hidden_size*2))
        low_intention = mlp_out.view(obs.size(0), obs.size(1), -1)
        # low_intention = mlp_out.view(obs.size(0), obs.size(1), -1).max(1)[0]
        return low_intention

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