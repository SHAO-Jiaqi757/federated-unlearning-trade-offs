"""
This example uses a very simple model and the MNIST dataset to show how the model,
the training and validation datasets, as well as the training and testing loops can
be customized in Plato.
"""
# from server import Server
from client import Client
from server import Server
from trainer import Trainer

def main():
    """
    A Plato federated learning training session using a custom model,
    datasource, and trainer.
    """
    client = Client(trainer=Trainer)
    server = Server(trainer=Trainer)
    server.run(client)


if __name__ == "__main__":
    main()
