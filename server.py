import asyncio
import math
import os
import pickle
from plato.servers import fedavg
from plato.config import Config
import logging
from plato.callbacks.server import ServerCallback

from plato.samplers import registry as samplers_registry

class Server(fedavg.Server):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        
        if hasattr(Config().server, "unlearn"):
            self.unlearn = Config().server.unlearn
            self.removed_clients = Config().server.removed_clients # list of client_ids to remove
            if hasattr(Config().server, "retrain"):
                self.retrain = Config().server.retrain
            else:
                self.retrain = False
        else:
            self.unlearn = False
        if hasattr(Config().server, "unlearn_strategy") and Config().server.unlearn_strategy == "tradeoff_fairness":
            self.lambdas = [0 for _ in range(self.total_clients)]
    def weights_received(self, weights_received):
        """
        Method called after the updated weights have been received.
        """
        client_ids = []
        num_samples = []
        for update in self.updates:
            client_ids.append(update.report.client_id)
            num_samples.append(update.report.num_samples)
        
        if self.current_round >= Config().trainer.rounds - 5: 
            checkpoints = {
                'client_ids': client_ids,
                'num_samples': num_samples,
                'weights': weights_received,
                'global_round': self.current_round,
            }
            
            # save the checkpoints
            checkpoint_dir = Config().params['checkpoint_path']
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_file = os.path.join(checkpoint_dir, f'checkpoints_{self.current_round}.pkl')
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoints, f)
         
        return weights_received
    
    def clients_processed(self) -> None: 
        if hasattr(Config().server, "unlearn_strategy") and Config().server.unlearn_strategy == "tradeoff_fairness":
            pretrained_local_accs = Config().trainer.pretrained_local_accs
            
            total_deltas = 1
            clients_count = 0
            self.l_accuracies = [0 for _ in range(self.total_clients)]
            logging.info(f"debugging: self.current_processed_clients: {self.current_processed_clients}")
            for client_id in range(1, self.total_clients+1):
                if client_id not in self.removed_clients:
                    clients_count += 1
                    testset_sampler = samplers_registry.get(self.datasource, client_id, testing=True)     
                    test_l =  self.trainer.test(self.testset, testset_sampler) * 100
                    logging.info(f"debugging: client_id {client_id} test_l - pretrained_local_acc: {test_l} - {pretrained_local_accs[client_id-1]} = {test_l - pretrained_local_accs[client_id-1]}")
                    delta_ = - test_l + pretrained_local_accs[client_id-1]
                    self.l_accuracies[client_id-1] = math.exp(delta_)
                    total_deltas += math.exp(delta_)
                    
            avg_delta = (total_deltas-1)/clients_count  
            for client_id in range(1, self.total_clients+1):
                if client_id not in self.removed_clients:
                    self.lambdas[client_id-1] = (self.l_accuracies[client_id-1]-avg_delta)/total_deltas
                    logging.info(f"debugging: client_id {client_id} lambda: {self.lambdas[client_id-1]}")        
    
    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        logging.info(f"self.unlearn: {self.unlearn}")
        if self.unlearn:
            self.unlearn_setup(updates)
            if not self.retrain and Config().server.unlearn_strategy == "tradeoff_stability":
                checkpoint_dir = Config().params['checkpoint_path']
                pretrained_gradients = []
            else: pretrained_gradients = None
        else: pretrained_gradients = None

        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples
            client_id = report.client_id
            
            # save the gradient from the last round of original training (resume round)
            if pretrained_gradients is not None and (client_id in self.removed_clients):
                with open(os.path.join(checkpoint_dir, f"pretrained_{client_id}.pkl"), "rb") as f:
                    pretrained = pickle.load(f)
                    if len(pretrained_gradients) == 0:
                        pretrained_gradients = [(num_samples / self.total_samples) * tensor.cpu().numpy() for tensor in pretrained["gradients"]]
                    else:
                        for i, tensor in enumerate(pretrained["gradients"]):
                            pretrained_gradients[i]+= (num_samples / self.total_samples) * tensor.cpu().numpy()
                    
                    
            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        if pretrained_gradients is not None:
            saved_data = {"gradients": pretrained_gradients}
            with open(os.path.join(checkpoint_dir, "pretrained.pkl"), "wb") as f:
                pickle.dump(saved_data, f)
        
        return avg_update
    
    def unlearn_setup(self, updates):
        """Assign 0 weights to removed clients."""
        if Config().server.retrain or \
            (not Config().server.retrain and self.current_round > 0):
            for update in updates:
                client_id = update.report.client_id
                if client_id in self.removed_clients:
                    logging.info(f"removed client {client_id}")
                    update.report.num_samples = 0
        
    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Customizes the server response with any additional information."""
        if hasattr(Config().server, "unlearn_strategy") and Config().server.unlearn_strategy == "tradeoff_fairness":
            server_response["lambdas"] = self.lambdas # for tradeoff_fairness
        return server_response 
    def _resume_from_checkpoint(self):
        """Resumes a training session from a previously saved checkpoint."""
        logging.info(
            "[%s] Resume a training session from a previously saved checkpoint.", self
        )

        checkpoint_path = Config.params["checkpoint_path"]
        # Loading important data in the server for resuming its session
        try:
            with open(f"{checkpoint_path}/current_round.pkl", "rb") as checkpoint_file:
                self.current_round = pickle.load(checkpoint_file)
        except:
            self.current_round = 0
            
                
        if hasattr(Config().server, "resume_round"):
            logging.info("Resuming from round %d", Config().server.resume_round)
            resume_round = Config().server.resume_round    
            if os.path.exists(os.path.join(Config().params['checkpoint_path'], "pretrained.pkl")):
                resume_round += 1 
            if resume_round+1 >= self.current_round and \
                    hasattr(Config().server, "resume_checkpoint_path"):
                checkpoint_path = Config().server.resume_checkpoint_path
        
            self.current_round = max(resume_round+1, self.current_round)-1

        self._restore_random_states(self.current_round, checkpoint_path)
        self.resumed_session = True

        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        filename = f"checkpoint_{model_name}_{self.current_round}.pth"
        

        self.trainer.load_model(filename, location=checkpoint_path)
        
        
        
class FUServerCallback(ServerCallback):
    def on_weights_aggregated(self, server, updates):
        ...
        # if hasattr(Config().server, "unlearn_strategy") and Config().server.unlearn_strategy == "tradeoff_fairness":
        #     # updating lambdas for tradeoff_fairness            
        #     sum_ = 1
        #     client_ids = []
        #     for update in updates:
        #         client_id = update.report.client_id
        #         if client_id in server.removed_clients:
        #             continue 
        #         client_ids.append(client_id)
        #         lambda_l = update.report.lambda_l
        #         logging.info(f"debugging: client_id {client_id} lambda_l: {lambda_l}")
        #         server.lambdas[client_id-1] = math.exp(lambda_l)
        #         sum_ += math.exp(lambda_l)
            
        #     avg_lambda = (sum_-1)/len(client_ids)
        #     for client_id in client_ids:
        #         server.lambdas[client_id-1] = (server.lambdas[client_id-1]-avg_lambda)/sum_
            
            
             
            
             
