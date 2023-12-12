from collections import Counter
import logging
import os
import pickle
from plato.clients import simple

from plato.config import Config


class Client(simple.Client):
    def _allocate_data(self) -> None:
        """Allocate training or testing dataset of this client."""
        if hasattr(Config().trainer, "use_mindspore"):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()

        
        self.targets = self.datasource.targets()
        self.num_train_examples = self.datasource.num_train_examples()
        counter = Counter(self.targets)
        
        # save client_id, num_samples, and class distribution
        
        checkpoint_dir = Config().trainer.checkpoint_dir if hasattr(Config().trainer, 'checkpoint_dir') \
            else "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f'client_info_{self.client_id}.pkl')
        
        logging.info(f"Saving client info to {checkpoint_file}")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'client_id': self.client_id,
                'sampler_condition': self.sampler.get_sampler_condition(),
                'num_samples': self.sampler.num_samples(),
            }, f)
            
    
    