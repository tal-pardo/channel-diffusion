import json
import argparse
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset

def get_config_pnp(config_main, config_car, config_ant, config_diff, config_noise):
    """
    Load five explicit config file paths for PnP use.
    Returns: configs, configs_car, configs_ant, configs_diff, configs_noise
    """
    with open(config_main, 'r') as f:
        configs = argparse.Namespace(**json.load(f))
    with open(config_car, 'r') as f:
        configs_car = argparse.Namespace(**json.load(f))
    with open(config_ant, 'r') as f:
        configs_ant = argparse.Namespace(**json.load(f))
    with open(config_diff, 'r') as f:
        configs_diff = argparse.Namespace(**json.load(f))
    with open(config_noise, 'r') as f:
        configs_noise = argparse.Namespace(**json.load(f))

    configs.BS_ant = torch.tensor(configs.BS_ant)
    if torch.cuda.is_available():
        configs.BS_ant = configs.BS_ant.cuda()

    return configs, configs_car, configs_ant, configs_diff, configs_noise


def load_dataset_deepmimo(args):
# def get_dataset(ratio: list, seed=1234, dataset_path='/mnt/HD2/yyz/MIMOlocdata32/', name='.'):
    dest_train = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_train.pt")
    dest_test = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_test.pt")
    dest_val = os.path.join(args.dataset_dir, args.dataset_file_name, "channel_val.pt")  # not always exist

    if os.path.exists(dest_train) and os.path.exists(dest_test):
        channel_train = torch.load(dest_train)
        channel_test = torch.load(dest_test)
        if os.path.exists(dest_val):
            channel_val = torch.load(dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]
    else:
        channel_file = os.path.join(args.dataset_dir, args.dataset_file_name, "data.npy")
        print("Loading channels", channel_file)

        channel = np.load(channel_file)  # N*1*ant*car complex
        channel_torch = torch.tensor(channel).squeeze(1)  # N*ant*car cfloat tensor
        num_data = channel_torch.shape[0]

        # Normalization
        channel_torch = channel_torch * 1e5

        perm = torch.randperm(num_data)

        num = int(args.ratio[0] * num_data)
        ids = perm[0:num]
        channel_train = channel_torch[ids]
        torch.save(channel_train, dest_train)

        num2 = int(args.ratio[1] * num_data)
        ids = perm[num:num+num2]
        channel_test = channel_torch[ids]
        torch.save(channel_test,  dest_test)

        if len(args.ratio) == 3:
            num3 = int(args.ratio[2] * num_data)
            ids = perm[num+num2:num+num2+num3]
            channel_val = channel_torch[ids]
            torch.save(channel_val,  dest_val)
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test),
                    TensorDataset(channel_val),]
        else:
            return [TensorDataset(channel_train),
                    TensorDataset(channel_test)]

def load_dataset(args):
    if args.dataset == "deepmimo":
        return load_dataset_deepmimo(args)
    else:
        raise NotImplementedError