import asyncio
import os
import pickle
from plato.servers import fedavg
from plato.config import Config
import logging

class Server(fedavg.Server):
    # def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
    #     super().__init__(model, datasource, algorithm, trainer, callbacks)
        
    #     if hasattr(Config().server, "unlearn"):
    #         self.unlearn = Config().server.unlearn
    #         self.removed_clients = Config().server.removed_clients # list of client_ids to remove
            
    #     else:
    #         self.unlearn = False

    def weights_received(self, weights_received):
        """
        Method called after the updated weights have been received.
        """
        client_ids = []
        num_samples = []
        for update in self.updates:
            client_ids.append(update.report.client_id)
            num_samples.append(update.report.num_samples)
            
        checkpoints = {
            'client_ids': client_ids,
            'num_samples': num_samples,
            'weights': weights_received,
            'global_round': self.current_round,
            'global_weights': self.algorithm.extract_weights()
        }
        
        # save the checkpoints
        checkpoint_dir = Config().trainer.checkpoint_dir if hasattr(Config().trainer, 'checkpoint_dir') \
            else "checkpoints"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoints_{self.current_round}.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoints, f)
        
        return weights_received
    
    
    
    
    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        # if self.unlearn:
        #     if hasattr(Config().server, "retrain"):
        #         self.retrain = Config().server.retrain
        #         if self.retrain:
        #             self.retrain_setup(updates)
                    
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update
    
    def retrain_setup(self, updates):
        for update in updates:
            client_id = update.report.client_id
            if client_id in self.removed_clients:
                update.report.num_samples = 0
            