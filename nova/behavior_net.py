import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
import math

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TrajectoryEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input, hidden):
        input = f.relu(self.linear(input))
        hidden = hidden.contiguous()
        _, final_h = self.gru(input, hidden)
        return final_h  # [num_layers, batch_size, hidden_size]

class MotionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MotionEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input, hidden):
        input = f.relu(self.linear(input))
        _, final_h = self.gru(input, hidden)
        return final_h

class TypeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TypeEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        return x[:, -1, :]

class MultiHeadAttn(nn.Module):
    def __init__(self, input_size, output_size, num_heads):
        super(MultiHeadAttn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        
        self.head_dim = input_size // num_heads
        self.embed_dim = self.head_dim * num_heads
        
        self.qkv = nn.Linear(input_size, 3 * self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, output_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Combined projection for Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, self.num_heads, self.head_dim), qkv)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.softmax(scores)
        
        # Apply attention to values
        context = torch.matmul(attn, v).view(batch_size, -1)
        
        # Final linear layer
        output = self.fc(context)

        return output
    
class Behavior_Encoder(nn.Module):
    """
    Behavior Encoder combining trajectory, motion, and type encoding.
    :input:
     - history: [batch_size, max_history_len, obs_dim];
     - encoder_hidden: [num_layers, batch_size, hidden_size*2];
    :output:
     - final_h: Tensor of shape (num_layers, batch_size, hidden_size*2);
     - final_h_out: Tensor of shape (batch_size, output_size);
    """
    def __init__(self, input_size, hidden_size=64,
        num_layers=1, output_size=8):
        super(Behavior_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_checkpoint = True

        self.traj_encoder = TrajectoryEncoder(input_size = input_size - 5, hidden_size=hidden_size, num_layers=num_layers)
        self.motion_encoder = MotionEncoder(input_size = 4, hidden_size=hidden_size, num_layers=num_layers)
        self.type_encoder = TypeEncoder(input_size = 3, hidden_size=hidden_size)
        self.attention = MultiHeadAttn(input_size=hidden_size * 3, output_size=output_size, num_heads=4)

    def forward(self, input, hidden):
        assert not torch.isnan(input).any(), "NaN detected in input"
        assert not torch.isnan(hidden).any(), "NaN detected in hidden"
        # input shape: [batch_size, max_history_len, obs_dim]
        trajectory = input[:, :, :-5]
        motion = input[:, :, -5: -1]
        vtype = input[:, :, -1]
        assert torch.all((vtype >= 0) & (vtype <= 2)), "Input contains values outside the expected range"

        # hidden shape: [num_layers, batch_size, hidden_size*2]
        traj_hidden = hidden[:, :, :self.hidden_size].contiguous()
        motion_hidden = hidden[:, :, self.hidden_size:].contiguous()

        traj_hidden_new = self.traj_encoder(trajectory, traj_hidden)
        motion_hidden_new = self.motion_encoder(motion, motion_hidden)
        vtype_out = self.type_encoder(vtype)

        final_h = torch.cat([traj_hidden_new, motion_hidden_new], dim=-1)  # [num_layers, batch_size, 2*hidden_size]
        final_h_out = self.attention(torch.cat([final_h[-1], vtype_out], dim=-1))
        return final_h, final_h_out

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        encoded_input = f.relu(self.linear(encoded_input))
        decoded_output, hidden = self.rnn(encoded_input, hidden)
        decoded_output = self.tanh(decoded_output)
        decoded_output = self.dropout(decoded_output)
        decoded_output = self.out(decoded_output)
        return decoded_output, hidden

class Behavior_Latent_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.5):
        super(Behavior_Latent_Decoder, self).__init__()

        self.decoder = DecoderRNN(input_size, hidden_size, output_size, num_layers, dropout)

    def forward(self, curr_history, prev_latent, hidden):
        # For agent_i, choose history_i [n_thread, max_vehicle_num, max_history_len, obs_dim]
        # Prev latent / Updated latent: [n_thread, max_vehicle_num, latent_dim]

        n_thread, max_vehicle_num, max_history_len, obs_dim = curr_history.shape
        _, _, latent_dim = prev_latent.shape

        latent = prev_latent.reshape(n_thread, max_vehicle_num, 1, latent_dim)
        latent = torch.tile(latent, (1, 1, max_history_len, 1))

        decoder_input = torch.cat([curr_history, latent], dim=-1).reshape(n_thread * max_vehicle_num, max_history_len, -1)

        # Output (Predicted history) [n_thread * max_vehicle_num, max_history_len, obs_dim]
        outputs, hidden = self.decoder(decoder_input, hidden)
        return outputs, hidden
