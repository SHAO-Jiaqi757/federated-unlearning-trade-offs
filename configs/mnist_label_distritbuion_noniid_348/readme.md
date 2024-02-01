# Balancing Fairness 

## Config:

- `trainer.pretrained_local_loss`: list of pretrained model's local loss on all clients.  
- `server.unlearn_strategy`: unlearn strategy `tradeoff_fairness`
- `parameters.Lambda`: the max L1 norm of $\lambda_i$ in the local objective function. 
