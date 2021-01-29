import torch

def bbox_loss(output,target,N_reg):
    x = torch.abs(target-output)
    loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))
    lamb = 1.
    loss =lamb*(loss.sum()) /N_reg
    return loss