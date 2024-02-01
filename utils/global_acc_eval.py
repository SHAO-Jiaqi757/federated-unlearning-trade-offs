# data cifar10
# model resnet18


# model_path


import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from plato.models.resnet import Model
from plato.models import registry as models_registry
from plato.samplers import registry as samplers_registry
from plato.datasources import registry as datasources_registry
from datasource_ import *
from plato.config import Config
from plato.trainers import loss_criterion
round = 30
unlearn_round = Config().trainer.rounds
model_name = Config().trainer.model_name
print("mdoel_name", model_name)
base_path = Config().general.base_path 
client_number = Config().clients.total_clients 
checkpoint_unlearn_path = Config().server.checkpoint_path
org_checkpoint_dir = checkpoint_unlearn_path.split("_unlearn")[0]
checkpoint_dir = os.path.join(base_path, org_checkpoint_dir)
print("orginal_checkpoint_path: ", checkpoint_dir)
removed_clients = Config().server.removed_clients
removed_ids = "".join(map(str, removed_clients))
checkpoints_unlearn_retrain_dir = os.path.join(base_path, checkpoint_unlearn_path)
print("unlearn_checkpoint_path: ", checkpoints_unlearn_retrain_dir)

loss_criterion_ = loss_criterion.get()

model = models_registry.get()
    
save_results_dir = f"compare_retrain_results/{checkpoint_unlearn_path}_round_{round}"
os.makedirs(save_results_dir, exist_ok=True)
if Config().data.datasource == "MNIST":
    datasets = MNISTDataSource(data_path=base_path+"/data")
elif Config().data.datasource == "CIFAR10":
    datasets = DataSource(data_path=base_path+"/data")

def load_model_weights(model_path):
    try:
        return torch.load(model_path)
    except:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
        
def test(model, test_loader, loss_criterion=None):
    model.eval()
    correct = 0
    total = 0
    class_acc = {}
    
    total_loss = 0 if loss_criterion is not None else None
    avg_loss = 0 if loss_criterion is not None else None
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            if loss_criterion is not None:
                loss_ = loss_criterion_(outputs, labels)
                total_loss += loss_.item() * labels.size(0)
                avg_loss = total_loss / total
            
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label not in class_acc:
                    class_acc[label] = [0, 0]
                class_acc[label][0] += 1
                if label == pred:
                    class_acc[label][1] += 1

    # sort highest acc
    class_acc = {k: v[1]/v[0]* 100 for k, v in sorted(class_acc.items(), key=lambda item: item[1][1]/item[1][0], reverse=True)}
    test_acc = 100 * correct / total
    return class_acc, test_acc, total, avg_loss

# plot histogram

def plot_class_acc(class_acc, test_acc):
    plt.figure(figsize=(10, 5))
    plt.bar(list(class_acc.keys()), list(class_acc.values()), color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(list(class_acc.keys()))
    plt.xlabel("class")
    plt.ylabel("accuracy")
    plt.title("class-wise accuracy (Tot Test Acc = %.2f %%)" % (100 * test_acc))
    plt.savefig(f"{save_results_dir}/class_acc_{model_name}_{round}.png")

avg_loss_clients = np.zeros(client_number)
def client_test_acc(client_id, org_model, unlearn_model):
    datasource = datasources_registry.get(client_id = client_id)
    testset = datasource.get_test_set()
    registered_sampler = samplers_registry.get(
            datasource, client_id, testing=True)
    # print(f"client_id = {client_id}, random_seed = {registered_sampler.random_seed}")  
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=10, sampler = registered_sampler.get())
 
    class_acc, test_acc, sample_size, avg_loss = test(org_model, test_loader, loss_criterion=loss_criterion_)
    print(f"client_id = {client_id}, avg_loss = {avg_loss}") 
    avg_loss_clients[client_id-1] = avg_loss
    class_acc_un, test_acc_un, sample_size, _ = test(unlearn_model, test_loader)
    return class_acc, test_acc, class_acc_un, test_acc_un, sample_size

def plot_class_acc_compare(class_acc, unlearn_class_acc, test_acc, unlearn_test_acc, label="Server"):
    plt.figure(figsize=(10, 5))
    # offsets two bars
    if label == "Server":
        color_1 = (0.2, 0.4, 0.6, 0.6)
        color_2 = (0.2, 0.4, 0.6, 0.2)
        print(f"per_class_acc (Original): {class_acc}")
        print(f"per_class_acc (Unlearn_): {unlearn_class_acc}")
    else:
        color_1 = (0.6, 0.4, 0.2, 0.6)
        color_2 = (0.6, 0.4, 0.2, 0.2)
    plt.bar(np.array(list(class_acc.keys())) - 0.2, list(class_acc.values()), width=0.4, color=color_1, label="original")
    plt.bar(np.array(list(unlearn_class_acc.keys())) + 0.2, list(unlearn_class_acc.values()), width=0.4, color=color_2, label="unlearn")
    plt.xticks(list(class_acc.keys()))
    plt.xlabel("class")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title(f"{label} class-wise accuracy (Tot Test Acc = %.2f %%, Unlearned Test Acc = %.2f %%)" % (100 * test_acc, 100 * unlearn_test_acc))
    plt.savefig(f"{save_results_dir}/{label}_class_acc_compare_{model_name}_{round}.png")


def compare_client_test_acc(client_id, org_model, unlearn_model):
    class_acc, test_acc, unlearn_class_acc, unlearn_test_acc, sample_size = client_test_acc(client_id, org_model, unlearn_model)
    plot_class_acc_compare(class_acc, unlearn_class_acc, test_acc, unlearn_test_acc, label=f"Client {client_id}")
    return class_acc, unlearn_class_acc, test_acc, unlearn_test_acc, sample_size


def main():
    test_loader = torch.utils.data.DataLoader(datasets.testset, batch_size=128, shuffle=False, num_workers=10)
    # original model
    original_model_path = os.path.join(checkpoint_dir, f'checkpoint_{model_name}_{round}.pth')
    model.load_state_dict(load_model_weights(original_model_path))
    class_acc, test_acc, _, _ = test(model, test_loader)
    # unlearn model
    unlearn_model_path = os.path.join(checkpoints_unlearn_retrain_dir, f'checkpoint_{model_name}_{unlearn_round}.pth')
    
    unlearn_model = models_registry.get()
    unlearn_model.load_state_dict(load_model_weights(unlearn_model_path))
    
    unlearn_class_acc, unlearn_test_acc, _, _ = test(unlearn_model, test_loader)
    
    plot_class_acc_compare(class_acc, unlearn_class_acc, test_acc, unlearn_test_acc)

    client_test_accs = np.zeros(client_number)
    client_test_accs_unlearn = np.zeros(client_number)
    
    # unlearn testing
    total_unlearn_test_samples = 0
    correctness_remaining = 0
    
    for client_id in range(1, client_number+1):
        _, _, l_test_acc, l_unlearn_test_acc, sample_size = compare_client_test_acc(client_id, model, unlearn_model) 
        client_test_accs[client_id-1] = l_test_acc
        client_test_accs_unlearn[client_id-1] = l_unlearn_test_acc
        
        if client_id not in removed_clients: 
            total_unlearn_test_samples += sample_size
            correctness_remaining += sample_size * l_unlearn_test_acc/100 
        else: 
            client_test_accs[client_id-1] = 0
            client_test_accs_unlearn[client_id-1] = 0
    print(f"unlearning acc: {correctness_remaining/total_unlearn_test_samples * 100: .2f}%")
    # print(f"avg_loss_clients: {avg_loss_clients}")
    print("client_test_accs: ", client_test_accs)
    # plot client_test_accs_unlearn - client_test_accs
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(1, client_number+1), client_test_accs_unlearn - client_test_accs)
    print("client_test_accs_unlearn - client_test_accs: ",
        client_test_accs_unlearn - client_test_accs)
    plt.xticks(np.arange(1, client_number+1))
    plt.xlabel("client id")
    plt.ylabel("Delta Test Acc (Unlearn - Original)")
    plt.title("Client-wise Delta Test Acc")
    print("saving to ", f"{save_results_dir}/client_wise_delta_test_acc_{model_name}_{round}.png")
    plt.savefig(f"{save_results_dir}/client_wise_delta_test_acc_{model_name}_{round}.png")
     

   

if __name__ == "__main__":
    main()