import os
import pickle
from plato.servers import fedavg
from plato.config import Config
import logging

class Server(fedavg.Server):

    def weights_received(self, weights_received):
        """
        Method called after the updated weights have been received.
        """
        client_ids = []
        num_samples = []
        for update in self.updates:
            client_ids.append(update.client_id)
            num_samples.append(update.num_samples)
            
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