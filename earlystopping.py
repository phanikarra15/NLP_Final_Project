#Class for Early Stopping Implementation
# Done by Srihasa, Phani, Mithilaesh

import torch
import numpy as np

class EarlyStopping:
    
    def __init__(self,path,patience,verbose,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.min_loss = np.Infinity
        self.counter = 0
    
    def __call__(self,validation_loss,model):
        validation_score = -validation_loss
        
        if self.best_score is None:
            self.best_score = validation_score
            self.save_checkpoint(validation_loss,model)
        
        elif validation_score < self.best_score + self.delta:
            self.counter +=  1
            print(f'Early Stopping = {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = validation_score
            self.save_checkpoint(validation_loss,model)
            self.counter = 0
            
    
    def save_checkpoint(self,validation_loss,model):
        if self.verbose is True:
            print(f'Validation loss is ({self.min_loss:.5f} --> {validation_loss:.5f}).  Saving model ...')
        torch.save(model, self.path)
        self.min_loss = validation_loss