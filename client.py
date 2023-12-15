from collections import Counter
import logging
import os
import pickle
import numpy as np
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
        
        # save client_id, num_samples, and class distribution
        
        checkpoint_dir = Config().params['checkpoint_path']
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f'client_info_{self.client_id}.pkl')
        
        # print all attributes of self.sampler
        if Config().data.sampler != "iid":
            if  hasattr(self.sampler, 'get_sampler_condition'):
                sampler_condition = self.sampler.get_sampler_condition()
            elif hasattr(self.sampler, 'get_trainset_condition'):
                logging.info("Sampler does not have get_sampler_condition method, using get_trainset_conditions instead")
                sampler_condition = self.sampler.get_trainset_condition() # samper_condition = np.asarray((unique, counts)).T
                # decompose sampler_condition = np.asarray((unique, counts)).T
                unique_classes = sampler_condition[:,0] # class_text 
                counts = sampler_condition[:,1]
                logging.info(f"unique_classes: {unique_classes}")
                # total number of samples
                num_samples = sum(counts)
                client_partition = num_samples/ len(self.sampler.targets_list)
                # class distribution
                classes_text_list = self.datasource.classes()
                client_label_proportions = np.zeros(len(classes_text_list))
                client_label_proportions[unique_classes] = counts/num_samples
         
                sampler_condition = (client_partition, client_label_proportions)
            else:
                raise ValueError("Sampler does not have get_sampler_condition or get_trainset_conditions method")
            logging.info(f"Saving client info to {checkpoint_file}")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'client_id': self.client_id,
                    'sampler_condition': sampler_condition,
                    'num_samples': self.sampler.num_samples(),
                }, f)
            
    
    