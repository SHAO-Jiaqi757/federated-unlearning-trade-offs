import logging
import os
import pickle
from plato.trainers import basic

from plato.config import Config
import torch

class Trainer(basic.Trainer):
    def train(self, trainset, *args) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        """
        if hasattr(Config().server, "removed_clients"):
            if hasattr(Config().server, "unlearn") and Config().server.unlearn:
                    
                if self.client_id in Config().server.removed_clients and self.current_round > Config().server.resume_round+1:
                    logging.info(f"Client {self.client_id} is removed, skip training")
                    return 0
        
        return super().train(trainset, *args)
        

    def test(self, testset, sampler=None):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """

        # Deactivate the cut layer so that testing uses all the layers
        if hasattr(Config().server, "removed_clients"):
            if hasattr(Config().server, "unlearn") and Config().server.unlearn:
                    
                if self.client_id in Config().server.removed_clients:
                    logging.info(f"Client {self.client_id} is removed, skip testing")
                    return 0
        
        return super().test(testset, sampler)


    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """
        self.optimizer.zero_grad()

        outputs = self.model(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()
        
        # unlearn 
        if hasattr(Config().server, "unlearn") and Config().server.unlearn:
            if hasattr(Config().server, "unlearn_strategy"):
                strategies = {
                    "tradeoff_stability": self.handle_tradeoff_stability,
                    "tradeoff_fairness": ...,
                    "tradeoff_all": ...,
                }
                
                strategies[Config().server.unlearn_strategy]()

        self.optimizer.step()

        return loss
    
    
    def handle_tradeoff_stability(self):
        
        self.resume_round = Config().server.resume_round if hasattr(Config().server, "resume_round") else 1 
        # logging.info(f"debugging, in {self.current_round}th round, resume_round: {self.resume_round}")
        
        if hasattr(Config().server, "resume_checkpoint_path"):
            resume_checkpoint_path = Config().server.resume_checkpoint_path
  
        if self.current_round <= self.resume_round + 1:
            # compute current gradient and save. 
            pretrained_gradients = [param.grad.clone() for param in self.model.parameters()]
            checkpoint_dir = Config().params['checkpoint_path']
            saved_data = {"gradients": pretrained_gradients}
            with open(os.path.join(checkpoint_dir, f"pretrained_{self.client_id}.pkl"), "wb") as f:
                pickle.dump(saved_data, f)
                
        else:
            # load pretrained gradients and params
            checkpoint_dir = Config().params['checkpoint_path']
            with open(os.path.join(checkpoint_dir, "pretrained.pkl"), "rb") as f:
                pretrained = pickle.load(f)
                pretrained_gradients = pretrained["gradients"]
                pretrained_params = load_model_weights(os.path.join(resume_checkpoint_path, f"checkpoint_{Config().trainer.model_name}_{self.resume_round+1}.pth"))
                
            # g_2 = pretrained_gradients + L (w - pretrained_params)
            L = 5
            a = 0.5 
            lambda_ = 0.1

            device = next(self.model.parameters()).device  # Get the device model is on
                        
            g_2 = [torch.tensor(g).to(device) for g in pretrained_gradients]
            for (g_, (param_name, w)) in zip(g_2, self.model.named_parameters()):
                w_hat = pretrained_params[param_name].to(device)
                g_ += L * (w - w_hat)
                
            
            for param, g in zip(self.model.parameters(), g_2):
                g_1 = param.grad
                proj_g_1_on_g2 = torch.dot(g_1.view(-1), g.view(-1)) / torch.dot(g.view(-1), g.view(-1)) * g
                param.grad = (g_1 - proj_g_1_on_g2) + a * lambda_ * g
            
            # logging.info(f"debugging, updated gradient in {self.current_round}th round")
            
            
            

def load_model_weights(model_path):
    try:
        return torch.load(model_path)
    except:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
        