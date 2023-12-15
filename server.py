import asyncio
import os
import pickle
from plato.servers import fedavg
from plato.servers.base import ServerEvents
from plato.config import Config
import logging
import socketio
from aiohttp import web
from plato.utils import s3, fonts


class Server(fedavg.Server):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        
        if hasattr(Config().server, "unlearn"):
            self.unlearn = Config().server.unlearn
            self.removed_clients = Config().server.removed_clients # list of client_ids to remove
            
        else:
            self.unlearn = False

    def weights_received(self, weights_received):
        """
        Method called after the updated weights have been received.
        """
        client_ids = []
        num_samples = []
        for update in self.updates:
            client_ids.append(update.report.client_id)
            num_samples.append(update.report.num_samples)
        
        if self.current_round >= Config().trainer.rounds - 10: 
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
    
    
    
    
    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        logging.info(f"self.unlearn: {self.unlearn}")
        if self.unlearn:
            if hasattr(Config().server, "retrain"):
                self.retrain = Config().server.retrain
                if self.retrain:
                    logging.info("retrain..")
                    self.retrain_setup(updates)
                    
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
                logging.info(f"removed client {client_id}")
                update.report.num_samples = 0
            
            
            
    def start(self, port=Config().server.port):
        """Starts running the socket.io server."""
        logging.info(
            "Starting a server at address %s and port %s.",
            Config().server.address,
            port,
        )

        self.sio = socketio.AsyncServer(
            ping_interval=self.ping_interval,
            max_http_buffer_size=2**31,
            ping_timeout=self.ping_timeout,
        )
        self.sio.register_namespace(MyServerEvents(namespace="/", plato_server=self))

        if hasattr(Config().server, "s3_endpoint_url"):
            self.s3_client = s3.S3()

        app = web.Application()
        self.sio.attach(app)
        web.run_app(
            app, host=Config().server.address, port=port, loop=asyncio.get_event_loop()
        )
        
        
class MyServerEvents(ServerEvents):
    
    def _resume_from_checkpoint(self):
        """Resumes a training session from a previously saved checkpoint."""
        logging.info(
            "[%s] Resume a training session from a previously saved checkpoint.", self
        )

        checkpoint_path = Config.params["checkpoint_path"]
        # Loading important data in the server for resuming its session
        if hasattr(Config().server, "resume_round"):
            resume_round = Config().server.resume_round    
        
        else:
            with open(f"{checkpoint_path}/current_round.pkl", "rb") as checkpoint_file:
                self.current_round = pickle.load(checkpoint_file)
                resume_round = self.current_round

        self._restore_random_states(self.current_round, checkpoint_path)
        self.resumed_session = True

        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        filename = f"checkpoint_{model_name}_{resume_round}.pth"
        
        if hasattr(Config().server, "resume_checkpoint_path"):
            checkpoint_path = Config().server.resume_checkpoint_path
        
        self.trainer.load_model(filename, checkpoint_path)