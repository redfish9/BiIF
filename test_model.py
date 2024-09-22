import torch
import torch.nn as nn
import math

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
        print("qkv ", qkv)
        q, k, v = map(lambda t: t.view(batch_size, self.num_heads, self.head_dim), qkv)
        print("q: ", q)
        print("k: ", k)
        print("v: ", v)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        print("scores ", scores)
        attn = self.softmax(scores)
        print("attn ", attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v).view(batch_size, -1)
        print("context ", context)
        
        # Final linear layer
        output = self.fc(context)

        return output

hidden_size = 32
batch_size = 4 * 11 * 10

x = torch.zeros(batch_size, hidden_size * 3)
attention_net = MultiHeadAttn(input_size=hidden_size*3,
                              output_size=8, num_heads=4)
att_out = attention_net(x)

print(att_out)