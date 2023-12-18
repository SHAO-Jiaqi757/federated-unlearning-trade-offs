# data cifar10
# model resnet18


# model_path


import os
import pickle
import numpy as np
import torch
from plato.models.resnet import Model
from plato.models import registry as models_registry
from plato.samplers import registry as samplers_registry
from plato.datasources import registry as datasources_registry
from cifar10 import DataSource
from plato.config import Config

round = 30
unlearn_round = 40
model_name = Config().trainer.model_name
base_path = Config().general.base_path 
checkpoint_unlearn_path = Config().server.checkpoint_path
org_checkpoint_dir = checkpoint_unlearn_path.split("_unlearn")[0]
checkpoint_dir = os.path.join(base_path, org_checkpoint_dir)
removed_ids = "".join(map(str, Config().server.removed_clients))
checkpoints_unlearn_retrain_dir = os.path.join(base_path, checkpoint_unlearn_path)

model = models_registry.get(model_type="torch_hub", model_name=model_name)
save_results_dir = f"compare_retrain_results/{checkpoint_unlearn_path}_round_{round}"
os.makedirs(save_results_dir, exist_ok=True)
datasets = DataSource(data_path="scratch/data")

def load_model_weights(model_path):
    try:
        return torch.load(model_path)
    except:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
        
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_acc = {}
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            
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
    return class_acc, test_acc

# plot histogram
import matplotlib.pyplot as plt
# diff color 

def plot_class_acc(class_acc, test_acc):
    plt.figure(figsize=(10, 5))
    plt.bar(list(class_acc.keys()), list(class_acc.values()), color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(list(class_acc.keys()))
    plt.xlabel("class")
    plt.ylabel("accuracy")
    plt.title("class-wise accuracy (Tot Test Acc = %.2f %%)" % (100 * test_acc))
    plt.savefig(f"{save_results_dir}/class_acc_{model_name}_{round}.png")


def client_test_acc(client_id, org_model, unlearn_model):
    datasource = datasources_registry.get(client_id = client_id)
    testset = datasource.get_test_set()
    registered_sampler = samplers_registry.get(
            datasource, client_id, testing=True)
    # print(f"client_id = {client_id}, random_seed = {registered_sampler.random_seed}")  
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=10, sampler = registered_sampler.get())
 
    class_acc, test_acc = test(org_model, test_loader)
    
    
    class_acc_un, test_acc_un = test(unlearn_model, test_loader)
    return class_acc, test_acc, class_acc_un, test_acc_un

def plot_class_acc_compare(class_acc, unlearn_class_acc, test_acc, unlearn_test_acc, label="Server"):
    plt.figure(figsize=(10, 5))
    # offsets two bars
    if label == "Server":
        color_1 = (0.2, 0.4, 0.6, 0.6)
        color_2 = (0.2, 0.4, 0.6, 0.2)
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
    class_acc, test_acc, unlearn_class_acc, unlearn_test_acc = client_test_acc(client_id, org_model, unlearn_model)
    plot_class_acc_compare(class_acc, unlearn_class_acc, test_acc, unlearn_test_acc, label=f"Client {client_id}")
    return class_acc, unlearn_class_acc, test_acc, unlearn_test_acc


def main():
    test_loader = torch.utils.data.DataLoader(datasets.testset, batch_size=128, shuffle=False, num_workers=10)
    # original model
    original_model_path = os.path.join(checkpoint_dir, f'checkpoint_{model_name}_{round}.pth')
    model.load_state_dict(load_model_weights(original_model_path))
    class_acc, test_acc = test(model, test_loader)
    # unlearn model
    unlearn_model_path = os.path.join(checkpoints_unlearn_retrain_dir, f'checkpoint_{model_name}_{unlearn_round}.pth')
    
    unlearn_model = models_registry.get(model_type="torch_hub", model_name=model_name)
    unlearn_model.load_state_dict(load_model_weights(unlearn_model_path))
    
    unlearn_class_acc, unlearn_test_acc = test(unlearn_model, test_loader)
    
    plot_class_acc_compare(class_acc, unlearn_class_acc, test_acc, unlearn_test_acc)

    client_test_accs = np.zeros(10)
    client_test_accs_unlearn = np.zeros(10)
    
    for client_id in range(1, 11):
        _, _, l_test_acc, l_unlearn_test_acc = compare_client_test_acc(client_id, model, unlearn_model) 
        client_test_accs[client_id-1] = l_test_acc
        client_test_accs_unlearn[client_id-1] = l_unlearn_test_acc
    
    # plot client_test_accs_unlearn - client_test_accs
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(1, 11), client_test_accs_unlearn - client_test_accs)
    plt.xticks(np.arange(1, 11))
    plt.xlabel("client id")
    plt.ylabel("Delta Test Acc (Unlearn - Original)")
    plt.title("Client-wise Delta Test Acc")
    print("saving to ", f"{save_results_dir}/client_wise_delta_test_acc_{model_name}_{round}.png")
    plt.savefig(f"{save_results_dir}/client_wise_delta_test_acc_{model_name}_{round}.png")
     

   

if __name__ == "__main__":
    main()