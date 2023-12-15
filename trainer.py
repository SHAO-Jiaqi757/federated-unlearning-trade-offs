import logging
from plato.trainers import basic

from plato.config import Config

class Trainer(basic.Trainer):
    def train(self, trainset, *args) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        """
        if hasattr(Config().server, "removed_clients"):
            if hasattr(Config().server, "unlearn") and Config().server.unlearn:
                    
                if self.client_id in Config().server.removed_clients:
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
