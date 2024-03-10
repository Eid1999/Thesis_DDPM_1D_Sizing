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

from MLP import MLP
from MLP_skip import MLP_skip
import optuna


class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-7, beta_end=0.02e-3, vect_size=91, device="cuda"):
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
        return torch.clip(betas, 0, 0.4999)
    
    def forward_process(self,x,t):
        # p(xt|x0) := N (xt; √αtx0, (1 − αtI))
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None]
        sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha[t])[:,None]
        epsilon=torch.rand_like(x)
        noisy_x = sqrt_alpha_hat*x + epsilon * sqrt_one_minus_alpha_hat
        return noisy_x,epsilon
    def sample_time(self,n):
        return torch.randint(0,self.noise_steps,size=(n,),device=self.device)
    def sampling(self,model,n,y,weight=3.0):
        model.eval()

        with torch.no_grad():
            x =     0.25*torch.randn(n, self.vect_size, 1, device=self.device, dtype=torch.float).squeeze()+.75
            for i in reversed(range(1,self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device) 
                alpha = self.alpha[t][:, None]
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
                sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha_hat[t])[:, None]
                beta=self.beta[t][:,None]
                z=.25*torch.randn(n, self.vect_size, 1, device=self.device, dtype=torch.float32).squeeze() if i>1 else torch.zeros_like(x)
                predicted_noise=model(x,t,y=y)
                if weight>0:
                    uncoditional_predicted_noise=model(x,t)
                    # predicted_noise=torch.lerp(uncoditional_predicted_noise,predicted_noise,weight)
                    predicted_noise=(1+weight)*predicted_noise -weight*uncoditional_predicted_noise
                x=1/torch.sqrt(alpha) *(x - ((1-alpha)/sqrt_one_minus_alpha_hat)*predicted_noise)+torch.sqrt(beta)*z
                x=x.clip(.5,1)
        model.train()

        
        return x
    def epoch_graph(self, X_train, X_test,y_train,y_test,hyper_parameters,epochs=200):
        
        train_losses,val_losses=self.reverse_process(X_train,y_train,epochs=epochs,val_x=X_test,val_y=y_test, **hyper_parameters,early_stop=True)
        epoch_LIST=torch.arange(1,len(train_losses)+1).cpu()
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
        
    def reverse_process(self,x,y,epochs=1000,batch_size=32,val_x=None,val_y=None,hidden_size1=80, hidden_size2=60,hidden_size3=40,hidden_size4=50,hidden_size5=50,learning_rate=0.001,early_stop=False):
        model=MLP(input_size=x.shape[1],output_size=x.shape[1],Y_dim=y.shape[-1],hidden_size1=hidden_size1, hidden_size2=hidden_size2,hidden_size3=hidden_size3,hidden_size4=hidden_size4,hidden_size5=hidden_size5).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
        Loss_Function=nn.MSELoss()
        # Loss_Function= self.MAPE
        loss=0
        dataloader=DataLoader(TensorDataset(x,y),batch_size=batch_size,shuffle=False)
        dataloader_val=[]
        if val_x is not None and val_y is not None:
            dataloader_val=DataLoader(TensorDataset(val_x,val_y),batch_size=batch_size,shuffle=False)
        epoch_loss_training= []
        epoch_loss_val=[] 
        training_loss=[]
        val_loss=[]         
        best_val_loss = float('inf')
        counter = 0
        for epoch in tqdm(range(epochs)):
                
            for i,(vector,y) in enumerate(dataloader):
                vector=vector.to(self.device)
                t=self.sample_time(vector.shape[0]).to(self.device)
                vector,noise=self.forward_process(vector,t)
                if np.random.random() < 0.1:
                    y = None
                predicted_noise=model(vector,t,y)
                loss=Loss_Function(predicted_noise,noise)
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
                            loss=Loss_Function(predicted_val_noise,val_noise)
                            epoch_loss_val.append(loss.item())
                        model.train()
            if val_x is not None:
                val_loss.append(epoch_loss_val[-1])
            training_loss.append(epoch_loss_training[-1])
            # if early_stop and val_x is not None:
            #     if training_loss[-1] < (best_val_loss+.001):
            #         best_val_loss = training_loss[-1]
            #         counter = 0
            #     else:
            #         counter += 1
            #         if counter >= 5:
            #             print(f'Early stop at epoch {epoch + 1}')
            #             break
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

class HYPERPARAMETERS_OPT:
    def __init__(self,X_train,Y_train,X_val,Y_val):
        self.X_train=X_train
        self.X_val=X_val
        self.Y_train=Y_train
        self.Y_val=Y_val
        study = optuna.create_study(direction='minimize')  # We want to minimize the objective function
        study.optimize(self.objective, n_trials=10)
        with open('best_hyperparameters.json', 'w') as file:
            json.dump(study.best_params, file)
    def objective(self,trial):
        params = {
        'hidden_size1': trial.suggest_int('hidden_size1', 20, 1000, log=True),
        'hidden_size2': trial.suggest_int('hidden_size2', 20, 1000, log=True),
        'hidden_size3': trial.suggest_int('hidden_size3', 20, 1000, log=True),
        'hidden_size4': trial.suggest_int('hidden_size4', 20, 1000,log=True),
        'hidden_size5': trial.suggest_int('hidden_size5', 20, 1000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1,log=True)
        }
        DDPM=Diffusion(vect_size=self.X_train.shape[1])
        loss,val_loss=DDPM.reverse_process(self.X_train,self.Y_train,epochs=20,val_x=self.X_val,val_y=self.Y_val,**params, early_stop=True)
        return loss[-1]

def load_FCA():
    GBW = 0
    GDC = 1
    PM  = 2
    POWER = 3
    CLOAD = 4
    Values=["1.0","1.5","2.5","3.5","4.0","4.5","5.0"]
    final_matrix_Y = np.empty((0, 5))
    final_matrix_X = np.empty((0, 19))
    for c_value in Values:
        data = loadmat('/media/ssd/Anot/MEEC/Tese/Code/data/FCA/different Cload POFs_v2/Cload'+c_value+'00000e-13/matlab.mat')
        X = np.zeros((100, 5))
        Y = np.array(data['archive3d'])[:,:,-1]
        X[:,GBW] = np.array(data['archive_bw'])[-1,:]
        X[:,GDC] = np.array(data['archive_gain'])[-1,:]
        X[:,PM] = np.array(data['archive_pm'])[-1,:]
        X[:,POWER] = np.array(data['archive_power'])[-1,:]
        
        
        X[:,CLOAD] = data['Cload'][0,0]
        final_matrix_X = np.concatenate((final_matrix_X, Y), axis=0)
        final_matrix_Y = np.concatenate((final_matrix_Y, X), axis=0)
    df_Y = pd.DataFrame(final_matrix_Y, columns=['GBW', 'GDC', 'PM','POWER','CLOAD'])
    df_X=pd.DataFrame(final_matrix_X, columns=['LM1', 'LM2', 'LM3', 'LM4', 'LM5', 'LM6', 'LM7', 'LM8', 'WM1', 'WM2', 'WM3', 'WM4', 
    'WM5', 'WM6', 'WM7', 'WM8', 'Vcm1','Vcm2','Rb' ])
    return df_Y,df_X

def dataset_info(df):
    
    print(df.shape)
    print(df.columns)
    print(df.describe().T)


class GUIDANCE_WEIGTH_OPT:
    def __init__(self,DDPM,Y_test,X_test):
        self.X_test=X_test.cpu().numpy()
        self.Y_test=Y_test
        self.DDPM=DDPM
        study = optuna.create_study(direction='minimize')  # We want to minimize the objective function
        study.optimize(self.objective, n_trials=50)
        with open('best_weight.json', 'w') as file:
            json.dump(study.best_params, file)
    def objective(self,trial):
        params = {
        'weight': trial.suggest_float('weight', .01, 100, log=True),
        }
        X_Sampled = self.DDPM.sampling(self.DDPM.model.cuda(), self.Y_test.shape[0],self.Y_test,**params).cpu().numpy()
    
        error =np.max(np.mean(np.abs(np.divide((self.X_test - X_Sampled), self.X_test)), axis=0))
        return error
  

def normalize_values(values,min_val=None,max_val=None):
    if min_val is  None:
        min_val = np.min(values,axis=0)
        max_val = np.max(values,axis=0)
        
    normalized_values = (values - min_val) / (max_val - min_val) * 0.5 + 0.5 
    
    return normalized_values,min_val,max_val
def reverse_normalize_values(normalized_values, original_min, original_max):
    reversed_values = (normalized_values - 0.5) * 2 * (original_max - original_min) + original_min
    return reversed_values
def augment_data(X, Y, target, repetition_factor = 10, scale=0.2):
    """
    Augments data with performance figures that are also meet by the same sizing.
    Augmentation is done by adding point wiht performance that is a rand*scale*mean(Y)
    Arguments:
    X -- Circuit target specifications (ANN inputs)
    Y -- Circuit desing variables (ANN outputs)
    target -- array {-1,1}^n_y that indicates if solutions are feasible bellow the original
              specificaton (-1) or above the original specificaton (1)
    repetition_factor -- number of repeated points
    scale -- size of the variation
    """
    # CAN repeat Y after scaling as they are not modified in the augmentation procedure
    Y_rep = np.repeat(Y, repetition_factor, axis=0)
    X_rep = np.repeat(X, repetition_factor, axis=0)
    m, n_x = X_rep.shape
    # -1 means that specifications with a smaller value are also meet by the design, e.g GDC
    #  1 means that specifications with a larger value are also meet by the design, e.g IDD
    target_scale = scale*np.mean(X, axis = 0)*target
    Y_rep = np.concatenate((Y, Y_rep), axis = 0)
    X_rep = np.concatenate((X, X_rep + np.random.rand(m,n_x)*target_scale), axis = 0)

    return (X_rep, Y_rep)
def Train_error(Y_train,DDPM,best_weight,X_train,df_X):
    Y_test=np.tile(Y_train[-1].cpu().numpy(), (1000,1))
    Y_test=torch.tensor(Y_test, dtype=torch.float32,device=DDPM.device)
        


    # start_time = time.time()
    X_Sampled = DDPM.sampling(DDPM.model.cuda(), Y_test.shape[0],Y_test,weight=best_weight)
    end_time = time.time()
    # Y_test=pd.DataFrame(Y_test.cpu().numpy(),columns=df_Y.columns)
    # Y_test.to_csv('Y_test.csv')
    df_Sampled=pd.DataFrame(X_Sampled.cpu().numpy(),columns=df_X.columns)
    X_test=np.array(np.tile(X_train[-1].cpu().numpy(), (1000,1)))
    X_test=pd.DataFrame(X_test,columns=df_X.columns)
    error =np.mean(np.abs(np.divide((X_test - df_Sampled), X_test, out=np.zeros_like(X_test), where=(X_test != 0))), axis=0)
    print(f'\n{error}')
def Test_error(Y_test,DDPM,best_weight,X_test,df_X,df_X_test):
    X_Sampled = DDPM.sampling(DDPM.model.cuda(), Y_test.shape[0],Y_test,weight=best_weight)
  
    df_Sampled=pd.DataFrame(X_Sampled.cpu().numpy(),columns=df_X.columns)
    error =np.mean(np.abs(np.divide((X_test - df_Sampled), X_test, out=np.zeros_like(X_test), where=(X_test != 0))), axis=0)
    print(f'\n{error}')
    plt.subplot(1, 2, 1)
    plt.imshow(df_Sampled, cmap='viridis', aspect='auto')
    # plt.colorbar()
    plt.title('VCOTA Sample Dataset')

    plt.subplot(1, 2, 2)
    plt.imshow(df_X_test, cmap='viridis', aspect='auto')
    plt.title('VCOTA test Dataset')
    plt.show()
def target_Predictions(min_y,max_y,df_Y,DDPM,df_X,best_weight,min_value,max_value,n_samples=100):
    Y_test=np.array(np.concatenate([np.tile([50, 300e-6,60e+6,65], (n_samples,1)),np.tile([40, 700e-6,150e+6,55], (n_samples,1)),np.tile([50, 150e-6,30e+6,65], (n_samples,1))]))
    Y_test=pd.DataFrame(Y_test,columns=df_Y.columns)
    Y_test.to_csv('Y_test.csv')
    
    Y_test,_,_=normalize_values(Y_test,min_val=min_y,max_val=max_y)
    Y_test=torch.tensor(Y_test.values, dtype=torch.float32,device=DDPM.device)
    X_Sampled = DDPM.sampling(DDPM.model.cuda(), Y_test.shape[0],Y_test,weight=best_weight)
    df_Sampled=pd.DataFrame(X_Sampled.cpu().numpy(),columns=df_X.columns)
    df_Sampled=reverse_normalize_values(df_Sampled,min_value, max_value)
    df_Sampled.to_csv('Sampled.csv')
    print(f"\nTarget1:\n{df_Sampled[:n_samples].describe().loc[['mean', 'std']].T}")
    print(f"\nTarget2:\n{df_Sampled[n_samples:n_samples*2].describe().loc[['mean', 'std']].T}")
    print(f"\nTarget3:\n{df_Sampled[n_samples*2:].describe().loc[['mean', 'std']].T}")


def main():
    hyper_parameters={
        'hidden_size1':60,
        'hidden_size2':40,
        'hidden_size3':60,
        'hidden_size4':80,
        'hidden_size5':80,
        'learning_rate':0.00001
    }
    best_weight=10
    # dataframe = pd.read_csv("data/vcota.csv")
    # df_X = dataframe[['w8','w6','w4','w10','w1','w0','l8','l6','l4','l10','l1','l0']]
    # df_Y = dataframe[['gdc','idd','gbw','pm']]
    
    
    # dataset_info(df_X)
    
    # dataset_info(df_Y)

    df_Y,df_X=load_FCA()
    # df_X[0:100].to_csv('DF_x.csv')
    Y_rep=df_Y.values
    X_rep=df_X.values
    
    X,min_value,max_value=normalize_values(X_rep)
    Y,min_y,max_y=normalize_values(Y_rep)
    
    # X=polynomial_features_X.fit_transform(X)
    # Y=polynomial_features_Y.fit_transform(Y)
    print(X.shape)
    print(Y.shape)
    
    
    X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2)
    X_val, X_test,Y_val,Y_test = train_test_split(X_test,Y_test, test_size=0.5)

    # Y_train,X_train = augment_data(Y_train,X_train,np.array([-1,1, -1, 0]), repetition_factor=10)


    DDPM=Diffusion(vect_size=X.shape[1])
    X=torch.tensor(X, dtype=torch.float32,device=DDPM.device)
    # Y=torch.tensor(Y, dtype=torch.float32,device=DDPM.device)

    X_train=torch.tensor(X_train, dtype=torch.float32,device=DDPM.device)
    Y_train=torch.tensor(Y_train, dtype=torch.float32,device=DDPM.device)
    X_val=torch.tensor(X_val, dtype=torch.float32,device=DDPM.device)
    Y_val=torch.tensor(Y_val, dtype=torch.float32,device=DDPM.device)
    Y_test=torch.tensor(Y_test, dtype=torch.float32,device=DDPM.device)

    # DDPM.see_noise_data(X)

    HYPERPARAMETERS_OPT(X_train,Y_train,X_val,Y_val)
    with open('best_hyperparameters.json', 'r') as file:
        hyper_parameters = json.load(file)
    
    
    

    # DDPM.epoch_graph(X_train, X_val,Y_train,Y_val,hyper_parameters,epochs=100)


    
    DDPM.reverse_process(X_train,Y_train,val_x=X_val,val_y=Y_val,epochs=1000,**hyper_parameters,early_stop=True)
    torch.save(DDPM.model.state_dict(), "MLP.pth")



    
    DDPM.model=MLP(input_size=X.shape[1],output_size=X.shape[1],Y_dim=Y.shape[-1],**hyper_parameters)
    DDPM.model.load_state_dict(torch.load("MLP.pth"))

    df_X_test=pd.DataFrame(X_test,columns=df_X.columns)

    



    GUIDANCE_WEIGTH_OPT(DDPM,Y_val,X_val)
    with open('best_weight.json', 'r') as file:
        best_weight = json.load(file)['weight']
    
        
        


    # start_time = time.time()
    X_Sampled = DDPM.sampling(DDPM.model.cuda(), Y_test.shape[0],Y_test,weight=best_weight)
    end_time = time.time()
    df_Sampled=pd.DataFrame(X_Sampled.cpu().numpy(),columns=df_X.columns)
    
    print("\n\n\nTrain Error")
    Train_error(Y_train,DDPM,best_weight,X_train,df_X)
    print("\n\n\nTest Error")
    Test_error(Y_test,DDPM,best_weight,X_test,df_X,df_X_test)
    # target_Predictions(min_y,max_y,df_Y,DDPM,df_X,best_weight,min_value,max_value)
    
    
    
    

 
    
    # Calculate mean squared error
    
        

if __name__ == '__main__':
    main()

    # Obter media e desvio padrao dos targets
    # UNET MLP
    
    
