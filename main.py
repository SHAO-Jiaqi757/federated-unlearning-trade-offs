"""
This example uses a very simple model and the MNIST dataset to show how the model,
the training and validation datasets, as well as the training and testing loops can
be customized in Plato.
"""
# from server import Server
from client import * 
from server import *
from trainer import *

def main():
    """
    A Plato federated learning training session using a custom model,
    datasource, and trainer.
    """
    client = Client(trainer=Trainer, trainer_callbacks=[FUTrainerCallback], callbacks=[FUClientCallback])
    server = Server(trainer=Trainer, callbacks=[FUServerCallback])
    server.run(client)


if __name__ == "__main__":
    main()
