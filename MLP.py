import torch
import torch.nn as nn
import math
class MLP(nn.Module):
    def __init__(self, input_size=91, hidden_size1=60, hidden_size2=40,hidden_size3=60, hidden_size4=60,hidden_size5=60,output_size=91,dim=6,Y_dim=15,learning_rate=0.001,weight_decay=0):
        super(MLP, self).__init__()
        self.dim = dim
        # self.layer0 = nn.Linear(input_size, input_size)
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = nn.ReLU()    
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.activation3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size3, hidden_size4)
        self.activation4 = nn.ReLU()
        self.layer5 = nn.Linear(hidden_size4, hidden_size5)
        self.activation5 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size5, output_size)
        sinu_pos_emb=SinusoidalPosEmb(dim)
        # sinu_y_embedding=SinusoidalPosEmb(Y_dim)
        self.time_embedding1 = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, input_size)
        )
        self.time_embedding2 = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, hidden_size1)
        )
        self.time_embedding3 = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, hidden_size2)
        )
        self.time_embedding4 = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, hidden_size3)
        )
        self.time_embedding5 = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, hidden_size4)
        )

        self.time_embedding6 = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, hidden_size5)
        )

        self.y_vectorization1 = nn.Sequential(
            # sinu_y_embedding,
            nn.Linear(Y_dim, 4*Y_dim),
            nn.GELU(),
            nn.Linear(4*Y_dim,input_size)
        )
        self.y_vectorization2 = nn.Sequential(
            # sinu_y_embedding,
            nn.Linear(Y_dim, 4*Y_dim),
            nn.GELU(),
            nn.Linear(4*Y_dim,hidden_size1)
        )
        self.y_vectorization3 = nn.Sequential(
            # sinu_y_embedding,
            nn.Linear(Y_dim, 4*Y_dim),
            nn.GELU(),
            nn.Linear(4*Y_dim,hidden_size2)
        )
        self.y_vectorization4 = nn.Sequential(
            # sinu_y_embedding,
            nn.Linear(Y_dim, 4*Y_dim),
            nn.GELU(),
            nn.Linear(4*Y_dim,hidden_size3)
        )
        self.y_vectorization5 = nn.Sequential(
            # sinu_y_embedding,
            nn.Linear(Y_dim, 4*Y_dim),
            nn.GELU(),
            nn.Linear(4*Y_dim,hidden_size4)
        )
        self.y_vectorization6 = nn.Sequential(
            # sinu_y_embedding,
            nn.Linear(Y_dim, 4*Y_dim),
            nn.GELU(),
            nn.Linear(4*Y_dim,hidden_size5)
        )
    def forward(self, x, t,y = None):
        t=t.unsqueeze(dim=-1).type(torch.float).squeeze()
        t1=self.time_embedding1(t)
        t2=self.time_embedding2(t)
        t3=self.time_embedding3(t)
        t4=self.time_embedding4(t)
        t5=self.time_embedding5(t)
        t6=self.time_embedding6(t)
        if y is not None:
            t1*=self.y_vectorization1(y)
            t2*=self.y_vectorization2(y)
            t3*=self.y_vectorization3(y)
            t4*=self.y_vectorization4(y)
            t5*=self.y_vectorization5(y)
            t6*=self.y_vectorization6(y)
        
        # x=self.layer0(x)
        x = self.layer1(x+t1)
        x = self.activation1(x)
        x = self.layer2(x+t2)
        x = self.activation2(x)
        x = self.layer3(x+t3)
        x = self.activation3(x)
        x = self.layer4(x+t4)
        x = self.activation4(x)
        x = self.layer5(x+t5)
        x = self.activation5(x)
        x = self.output_layer(x+t6)
        return x
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
