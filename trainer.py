import logging
import os
import pickle
from plato.trainers import basic

from plato.config import Config
import torch

from plato.callbacks.trainer import TrainerCallback

class FUTrainerCallback(TrainerCallback):
    # def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
    #     """
    #     Event called at the end of a training epoch.
    #     """
    #     if trainer.current_epoch==1 and batch ==0 and hasattr(Config().server, "unlearn_strategy") and Config().server.unlearn_strategy == "tradeoff_fairness":
    #         trainer._loss_tracker.g_loss = trainer._loss_tracker._average/ (1+trainer.lambda_l)
    #         logging.info("debugging: (in trainer) trainer.g_loss: %f", trainer._loss_tracker.g_loss)  
                
    def on_train_run_end(self, trainer, config, **kwargs):   
        # unlearn stability strategy
        if hasattr(Config().server, "unlearn") and Config().server.unlearn:
            if hasattr(Config().server, "unlearn_strategy") \
                    and Config().server.unlearn_strategy == "tradeoff_stability":    
                self.handle_tradeoff_stability(trainer)
    
    def handle_tradeoff_stability(self, trainer):
        
        self.resume_round = Config().server.resume_round if hasattr(Config().server, "resume_round") else 1 
        # logging.info(f"debugging, in {self.current_round}th round, resume_round: {self.resume_round}")
        
        removed_clients = Config().server.removed_clients
        if hasattr(Config().server, "resume_checkpoint_path"):
            resume_checkpoint_path = Config().server.resume_checkpoint_path

        logging.info(f"current_round: { trainer.current_round}; resume_round: {self.resume_round}")
        if trainer.current_round <= self.resume_round + 1:
            if trainer.client_id in removed_clients: 
                # compute current gradient and save. 
                pretrained_gradients = [param.grad.clone() for param in trainer.model.parameters()]
                checkpoint_dir = Config().params['checkpoint_path']
                saved_data = {"gradients": pretrained_gradients}
                with open(os.path.join(checkpoint_dir, f"pretrained_{trainer.client_id}.pkl"), "wb") as f:
                    pickle.dump(saved_data, f)
                    logging.info(f"saving pretrained_model of {trainer.client_id}")
                
        else:
            # load pretrained gradients and params
            checkpoint_dir = Config().params['checkpoint_path']
            with open(os.path.join(checkpoint_dir, "pretrained.pkl"), "rb") as f:
                pretrained = pickle.load(f)
                pretrained_gradients = pretrained["gradients"]
                pretrained_params = load_model_weights(os.path.join(resume_checkpoint_path, f"checkpoint_{Config().trainer.model_name}_{self.resume_round+1}.pth"))
                
            # g_2 = pretrained_gradients + L (w - pretrained_params)
            L = 5
            P = 0.3 
            lambda_ = Config().parameters.lambda_ if hasattr(Config().parameters, "lambda_") else 0.1 

            device = next(trainer.model.parameters()).device  # Get the device model is on
                        
            trainer.model.train()

            # Gradient correction
            g_c = [torch.tensor(g).to(device) for g in pretrained_gradients]
            for (g_, (param_name, w)) in zip(g_c, trainer.model.named_parameters()):
                w_hat = pretrained_params[param_name].to(device)
                g_ += L * (w - w_hat)

            for param, g in zip(trainer.model.parameters(), g_c):
                g_s = param.grad
                if g_s is not None:
                    proj_gc_on_gs = torch.dot(g.view(-1), g_s.view(-1)) / torch.dot(g_s.view(-1), g_s.view(-1)) * g_s
                    g_cor = g - proj_gc_on_gs
                    param.grad = (1 + lambda_ * (1 - P)) * g_s + P * lambda_ * g_cor

            # Update model parameters
            trainer.optimizer.step()

            # Optionally, zero the gradients after the update
            trainer.optimizer.zero_grad()


        
class Trainer(basic.Trainer):
    def train(self, trainset, *args) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        """
        self.resume_round = Config().server.resume_round if hasattr(Config().server, "resume_round") else self.current_round-2
        if hasattr(Config().server, "unlearn") and Config().server.unlearn:
                
            if self.client_id in Config().server.removed_clients and self.current_round > self.resume_round+1:
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
    

            # logging.info(f"debugging, updated gradient in {self.current_round}th round")
            

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
        
        if hasattr(Config().server, "unlearn") and Config().server.unlearn:
            if hasattr(Config().server, "unlearn_strategy") and Config().server.unlearn_strategy == "tradeoff_fairness":
                if not hasattr(self, "lambda_l"): self.lambda_l = 0
                loss = (1+ self.lambda_l) * loss
                
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss  
            

def load_model_weights(model_path):
    try:
        return torch.load(model_path)
    except:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
        