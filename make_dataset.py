import pandas as pd
from config import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import *
from dataset import *
from train import *
from args import *
import os
import pickle


if __name__ == '__main__':
    # consist data
    args = get_args()
    setSeeds()
    dataset = DataPreprocess(args) # 객체임
    
    os.makedirs("pkl", exist_ok=True)
    with open("pkl/" + args.dataset_path, "wb") as pkl:
        pickle.dump(dataset, pkl)