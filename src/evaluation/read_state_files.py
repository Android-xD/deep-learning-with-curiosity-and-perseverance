from src.utils.configs import trained_weights_dir
import os
import torch

if __name__ == '__main__':
    """Print the hyperparameters of the trained models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = os.listdir(trained_weights_dir)
    for model in models:
        state_dict = torch.load(os.path.join(trained_weights_dir, model), map_location=torch.device(device))
        if "model_state" in state_dict.keys():
            hyperparams = ["input_size", "batch_size", "num_epochs", "learning_rate", "dataset_name"]
            # print the hyperparameters
            print(",\t ".join([f" {k}: {state_dict[k]}" for k in hyperparams]) + "\t\t" + model)
            # extract num epochs from
            if "log" in state_dict.keys() and len(state_dict['log']) > 0:
                last_epoch = max([val['Epoch'] for val in state_dict['log']])
                # print the dictionary of the last epoch
                for i, val in enumerate(state_dict['log']):
                    if val['Epoch'] == last_epoch:
                        print(state_dict['log'][i])
                        break



