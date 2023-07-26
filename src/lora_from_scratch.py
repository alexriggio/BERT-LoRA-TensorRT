import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLoRA(nn.Module):
    """
    A low-rank adapted linear layer. 

    Args:
        in_dim: int = An integer representing the input dimension of the linear layer 
        out_dim: int = An integer representing the output dimension of the linear layer
        r: int = An integer representing the rank of the low-rank approximated matrices
        lora_alpha: int = An integer representing the numerator of the scaling constant alpha / r 
        lora_dropout: float = A float between 0 and 1 representing the dropout probability      
    """       
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.,    
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) 
        
        # Check that the rank is at least 1
        assert r > 0, "Variable 'r' is not greater than zero. Choose a rank of 1 or greater."
            
        # recreate the linear layer and freeze it (the actual weight values will be copied in outside of this class)
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False

        # create the low-rank A matrix and initialize with same method as in Hugging Face PEFT library
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # create the low-rank B matrix and initialize to zero 
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        nn.init.constant_(self.lora_B.weight, 0)

        # scaling constant
        self.scaling = self.lora_alpha / self.r
                        
    def forward(self, x):
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling        
        return pretrained_out + lora_out
    
    
def freeze_model(model):
    """Freezes all layers except the LoRa modules and classifier."""
    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False

            
def create_lora(module, r, lora_dropout, lora_alpha):
    """Converts a linear module to a LoRA linear module."""
    k, d = module.weight.shape  # pytorch nn.Linear weights are transposed, that is why shape is (k, d) and not (d, k)
    lora = LinearLoRA(d, k, r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
    with torch.no_grad():
        lora.pretrained.weight.copy_(module.weight)
        lora.pretrained.bias.copy_(module.bias)        
    return lora
                

def add_lora_layers(
    model,  
    module_names: Tuple=("query", "value"), 
    r: int=8, 
    lora_alpha: float=16,
    lora_dropout: float=0.1, 
    ignore_layers: List[int]=[]  
):
    """
        Replaces chosen linear modules with LoRA equivalents. 
     
        Args:
            model: torch.nn.Module = The PyTorch model to be used
            module_names: Tuple = A tuple containing the names of the linear layers to replace
                Ex. ("query") to replace the linear modules with "query" in the name --> bert.encoder.layer.0.attention.self.query
            r: int = 
            lora_alpha: int = An integer representing the numerator of the scaling constant alpha / r 
            lora_dropout: float = A float between 0 and 1 representing the dropout probability
            ignore_layers: list = A list with the indices of all BERT layers NOT to add LoRA modules 
        """                     
    module_types: Tuple=(nn.Linear,)
    
    # disable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if isinstance(module, module_types) and name in module_names:
            temp_lora = create_lora(module, r=r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
            setattr(model, name, temp_lora)                  
        else:
            ignore_layers_str = [str(i) for i in ignore_layers]
            if name not in ignore_layers_str:
                add_lora_layers(module, module_names, r, lora_dropout, lora_alpha, ignore_layers)
                
                
def unfreeze_model(model):
    """Unfreezes all parameters in a model by setting requires_grad to True."""
    for name, param in model.named_parameters():
        param.requires_grad = True

        
def create_linear(module):
    """Converts a LoRA linear module back to a linear module."""
    k, d = module.pretrained.weight.shape  # pytorch nn.Linear weights are transposed, that is why variables are k, d and not d, k
    linear = nn.Linear(d, k, bias=True)
    
    with torch.no_grad():
        linear.weight.copy_(module.pretrained.weight + (module.lora_B.weight @ module.lora_A.weight) * module.scaling)
        linear.bias.copy_(module.pretrained.bias)
        
    return linear


def merge_lora_layers(model, module_names: Tuple=("query", "value"), dropout=0.1):
    """
        Replaces LoRA modules with their original linear equivalents. 
   
        Args:
            model: torch.nn.Module = The PyTorch model to be used
            module_names: Tuple = A tuple containing the names of the LoRA layers to replace
                Ex. ("query") to replace the LoRA modules with "query" in the name --> bert.encoder.layer.0.attention.self.query
            r: int = 
            dropout: float = A float between 0 and 1 representing the dropout probability    
        """                     
    # enable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if name in module_names and hasattr(module, "pretrained"):
            temp_linear = create_linear(module)
            setattr(model, name, temp_linear)                  
        else:
            merge_lora_layers(module, module_names=module_names, dropout=0.1)
                         