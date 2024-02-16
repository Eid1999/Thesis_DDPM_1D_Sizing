import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures,Normalizer,StandardScaler,MinMaxScaler,StandardScaler
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import time
import itertools
import json
import torch.nn.functional as F




class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-7, beta_end=0.02e-3, vect_size=91, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.vect_size = vect_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        # self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)        

  
    def prepare_noise_schedule(self,s=0.008):

        #linear schedule

        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

        #cosine schedule

        x = torch.linspace(0, self.noise_steps, self.noise_steps+1)
        alphas_cumprod = (torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.9999)
    
    def forward_process(self,x,t):
        # p(xt|x0) := N (xt; √αtx0, (1 − αtI))
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None]
        sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha[t])[:,None]
        epsilon=torch.rand_like(x)
        noisy_x = sqrt_alpha_hat*x + epsilon * sqrt_one_minus_alpha_hat
        return noisy_x,epsilon
    def sample_time(self,n):
        return torch.randint(0,self.noise_steps,size=(n,),device=self.device)
    def sampling(self,model,n,y,weight=5):
        model.eval()

        with torch.no_grad():
            x = torch.randn(n, self.vect_size, 1, device=self.device, dtype=torch.float).squeeze()
            for i in tqdm(reversed(range(1,self.noise_steps)),position=1):
                t = (torch.ones(n) * i).long().to(self.device) 
                alpha = self.alpha[t][:, None]
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
                sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha_hat[t])[:, None]
                beta=self.beta[t][:,None]
                z= torch.randn(n, self.vect_size, 1, device=self.device, dtype=torch.float32).squeeze() if i>0 else torch.zeros_like(x)
                predicted_noise=model(x,t,y=y)
                if weight>0:
                    uncoditional_predicted_noise=model(x,t)
                    predicted_noise=torch.lerp(uncoditional_predicted_noise,predicted_noise,weight)
                x=1/torch.sqrt(alpha) *(x - ((1-alpha)/sqrt_one_minus_alpha_hat)*predicted_noise)+torch.sqrt(beta)*z
                x=x.clip(-1,1)
        model.train()

        
        return x
    def epoch_graph(self,X,y,hyper_parameters,epochs=200):
        X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
        epoch_LIST=torch.arange(1,epochs+1).cpu()
        train_losses,val_losses=self.reverse_process(X_train,y_train,epochs=epochs,val_x=X_test,val_y=y_test, **hyper_parameters)
        plt.figure()
        plt.plot(epoch_LIST, train_losses, 'b', label='Training Loss')
        plt.plot(epoch_LIST, val_losses, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
    def MAPE(self,predicted_noise,noise):
        return torch.mean(torch.abs((torch.div(predicted_noise - noise, noise,rounding_mode="trunc"))))
    def reverse_process(self,x,y,epochs=100,batch_size=32,val_x=None,val_y=None,hidden_size1=60, hidden_size2=40,hidden_size3=60):
        model=MLP(input_size=x.shape[1],output_size=x.shape[1],Y_dim=y.shape[-1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        mse=nn.MSELoss()
        loss=0
        dataloader=DataLoader(TensorDataset(x,y),batch_size=batch_size,shuffle=False)
        dataloader_val=[]
        if val_x is not None and val_y is not None:
            dataloader_val=DataLoader(TensorDataset(val_x,val_y),batch_size=batch_size,shuffle=False)
        epoch_loss_training= []
        epoch_loss_val=[] 
        training_loss=[]
        val_loss=[]         
  
        for epoch in tqdm(range(epochs)):
                
            for i,(vector,y) in enumerate(dataloader):
                vector=vector.to(self.device)
                t=self.sample_time(vector.shape[0]).to(self.device)
                vector,noise=self.forward_process(vector,t)
                if np.random.random() < 0.1:
                    y = None
                predicted_noise=model(vector,t,y)
                loss=mse(predicted_noise,noise) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_training.append(loss.item())
                if val_x is not None:
                    if len(dataloader_val)>i:
                        data_iterator = iter(dataloader_val)
                        model.eval()
                        with torch.no_grad():
                            val_vector,y_val=next(data_iterator)
                            val_vector=val_vector.to(self.device)
                            y_val=y_val.to(self.device)
                            t=self.sample_time(val_vector.shape[0]).to(self.device)
                            val_vector,val_noise=self.forward_process(val_vector,t)
                            predicted_val_noise=model(vector,t,y_val)
                            loss=mse(predicted_val_noise,val_noise)
                            epoch_loss_val.append(loss.item())
                        model.train()
            if val_x is not None:
                val_loss.append(epoch_loss_val[-1])
            training_loss.append(epoch_loss_training[-1])
        self.model=model
            
        return training_loss,val_loss



                
    def see_noise_data(self,x):

        noise_vect,_=self.forward_process(x,torch.full((x.shape[0],), self.noise_steps-1, device=self.device))

    # Iterate over each vector tensor and plot it
        original_matrix = x.cpu().squeeze().numpy()
        matrix_with_noise_array = noise_vect.cpu().squeeze().numpy()

        # Create a blank plot
        # fig, ax = plt.subplots()

        # Plot the original vector
        plt.subplot(1, 2, 1)
        plt.imshow(original_matrix, cmap='viridis', aspect='auto')
        # plt.colorbar()
        plt.title('VCOTA Initial Dataset')

        plt.subplot(1, 2, 2)
        plt.imshow(matrix_with_noise_array, cmap='viridis', aspect='auto')
        plt.title('VCOTA Noisy Dataset')

        plt.show()
    

    def grid_search(self,x,y):
        param_grid = {
            'hidden_size1': [10,30,60,120],
              'hidden_size2': [10,30,60,120],
              'hidden_size3': [10,30,60,120]
            # Add other hyperparameters
            }
        X_train, X_val,Y_train,Y_val = train_test_split(x,y, test_size=0.2)
        all_params = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
        val_loss=10
        best_params = 'NULL'
        for params in all_params:
            print(f'Testing Hyperparemeters{params}')
            _,vl=self.reverse_process(X_train,Y_train,epochs=50,val_x=X_val,val_y=Y_val,**params)
            if val_loss>vl[-1]:
                val_loss=vl[-1]
                best_params=params
            print(f'validation loss: {val_loss}')
        print(f'best value:{best_params}')
        with open('best_hyperparameters.json', 'w') as file:
            json.dump(best_params, file)
        return best_params



        



            
class MLP(nn.Module):
    def __init__(self, input_size=91, hidden_size1=80, hidden_size2=60,hidden_size3=40,hidden_size4=60,hidden_size5=80,output_size=91,dim=6,Y_dim=15):
        super(MLP, self).__init__()
        self.dim = dim
        # self.layer0 = nn.Linear(input_size, input_size)
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)
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
            t1+=self.y_vectorization1(y)
            t2+=self.y_vectorization2(y)
            t3+=self.y_vectorization3(y)
            t4+=self.y_vectorization4(y)
            t5+=self.y_vectorization5(y)
            t6+=self.y_vectorization6(y)
        
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

def load_FCA():
    GBW = 0
    GDC = 1
    PM  = 2
    POWER = 3
    CLOAD = 4
    Values=["1.0","1.5","2.5"]
    final_matrix = np.empty((0, 5))
    for c_value in Values:
        data = loadmat('/media/ssd/Anot/MEEC/3 sem/PIC/Code/data/FCA/different Cload POFs_v2/Cload'+c_value+'00000e-13/matlab.mat')
        X = np.zeros((100, 5))
        Y = np.array(data['archive3d'])[:,:,-1]
        X[:,GBW] = np.array(data['archive_bw'])[-1,:]
        X[:,GDC] = np.array(data['archive_gain'])[-1,:]
        X[:,PM] = np.array(data['archive_pm'])[-1,:]
        X[:,POWER] = np.array(data['archive_power'])[-1,:]
        
        
        X[:,CLOAD] = data['Cload'][0,0]
        final_matrix = np.concatenate((final_matrix, X), axis=0)
    df_Y = pd.DataFrame(final_matrix, columns=['GBW', 'GDC', 'PM','POWER','CLOAD'])
    return df_Y

def dataset_info(df):
    
    print(df.shape)
    print(df.columns)
    print(df.describe().T)



def main():
    hyper_parameters={
        'hidden_size1':60,
        'hidden_size2':40,
        'hidden_size3':60,
    }
    dataframe = pd.read_csv("data/vcota.csv")
    df_X = dataframe[['w8','w6','w4','w10','w1','w0','l8','l6','l4','l10','l1','l0']]
    df_Y = dataframe[['gdc','idd','gbw','pm']]


    # dataset_info(df_X)
    
    # dataset_info(df_Y)



    polynomial_features_X=PolynomialFeatures(2)
    polynomial_features_Y=PolynomialFeatures(2)
    
    scaler_x=MinMaxScaler(feature_range=(-1, 1))
    X=scaler_x.fit_transform(df_X)
    Y=MinMaxScaler(feature_range=(-1, 1)).fit_transform(df_Y)
    
    X=polynomial_features_X.fit_transform(X)
    Y=polynomial_features_Y.fit_transform(Y)
    print(X.shape)
    print(Y.shape)
    

    with open('best_hyperparameters.json', 'r') as file:
        hyper_parameters = json.load(file)
    X, X_test,Y,y_test = train_test_split(X,Y, test_size=0.2)
    DDPM=Diffusion(vect_size=X.shape[1])
    X=torch.tensor(X, dtype=torch.float32,device=DDPM.device)
    Y=torch.tensor(Y, dtype=torch.float32,device=DDPM.device)
    Y_test=torch.tensor(y_test, dtype=torch.float32,device=DDPM.device)

    # DDPM.grid_search(X,Y)
    
    # DDPM.see_noise_data(X)

    DDPM.epoch_graph(X,Y,hyper_parameters,epochs=50)



    DDPM.reverse_process(X,Y,epochs=50, **hyper_parameters)
    torch.save(DDPM.model.state_dict(), "MLP.pth")



    
    DDPM.model=MLP(input_size=X.shape[1],output_size=X.shape[1],Y_dim=Y.shape[-1])
    DDPM.model.load_state_dict(torch.load("MLP.pth"))
    start_time = time.time()
    X_Sampled = DDPM.sampling(DDPM.model.cuda(), Y_test.shape[0],Y_test)
    end_time = time.time()


    # print((end_time - start_time))
    # plt.figure()
    # plt.imshow(sampled_vect.cpu(), cmap='viridis',aspect='auto')
    # plt.colorbar()  # Add colorbar to show values' scale
    # plt.savefig('heatmap.png')
    # plt.show()

    X_test=X_test[:,1:df_X.shape[1]+1]
    X_test=scaler_x.inverse_transform(X_test)
    df_X_test=pd.DataFrame(X_test,columns=df_X.columns)

    X_Sampled=X_Sampled[:,1:df_X.shape[1]+1]
    X_Sampled=scaler_x.inverse_transform(X_Sampled.cpu().numpy())
    df_Sampled=pd.DataFrame(X_Sampled,columns=df_X.columns)
    # dataset_info(df_Sampled)
    df_Sampled.to_csv('Sampled.csv')
    error =np.mean(np.abs(((df_X_test - df_Sampled) / df_X_test)), axis=0)
    
    print(error)
    
    # Calculate mean squared error
    
    
    

if __name__ == '__main__':
    main()

    # https://scikit-learn.org/\stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html