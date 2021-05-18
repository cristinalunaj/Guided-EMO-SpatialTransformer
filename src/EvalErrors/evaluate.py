from __future__ import print_function
import argparse
import os
import sys

import pandas as pd

sys.path.append("../..")
sys.path.append("~/PycharmProjects/Guided-EMO-SpatialTransformer")
sys.path.append("~/PycharmProjects/Guided-EMO-SpatialTransformer/src")


import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from src.utils.plotcm import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_loaders.data_loaders import Plain_Dataset
from src.data_loaders.data_loader_land import Plain_Dataset_land
from src.data_loaders.data_loader_saliency import Plain_Dataset_saliency

from src.architectures.deep_emotion_saliency import Deep_Emotion_Saliency as Deep_Emotion_saliency_48x48
from src.architectures.deep_emotion_original import Deep_Emotion_Original as Deep_Emotion_Original_48x48
from src.architectures.deep_emotion_baseline import Deep_Emotion_Baseline as Deep_Emotion_Baseline_48x48

from src.utils.args_utils import str2bool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_weigths(root_path_weights, fold=-1):
    list_of_possible_weights = os.listdir(root_path_weights)
    df_path_weights = pd.DataFrame(list_of_possible_weights, columns=["path"])
    df_path_weights[["tmp", "deepEmo", "totalEpochs", "bs", "lr", "fold", "imgSize", "currentEpoch", "modality"]] = \
        df_path_weights["path"].str.split("-", -2, expand=True)
    #Remove wrong rows:
    df_path_weights = df_path_weights.loc[df_path_weights["modality"].notna()]
    if(fold<0): #Load complete model:
        fold = "COMPLETE"

    df_complete_models = df_path_weights.loc[df_path_weights["fold"] == str(fold)]
    df_complete_models["currentEpoch"] = pd.to_numeric(df_complete_models["currentEpoch"])
    df_complete_models_new = df_complete_models.sort_values(by="currentEpoch", ascending=False)

    #Get first row (last top model)
    try:
        top_path = df_complete_models_new["path"].iloc[0]
    except:
        print("ERROR!! There is no model for fold: ", str(fold))
        top_path = -1
    return top_path





def eval_5CV(k_folds, batch_size, root_path_weights, modality, logs_path):
    for fold in range(0, k_folds):
        #Select data to test
        test_ids = np.array(list(test_dataset.df_file.loc[test_dataset.df_file["fold"] == fold].index))
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        testloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=0,pin_memory=True)
        #Load weights
        weigths_path = get_weigths(root_path_weights, fold)
        print("Loaded weight: ", weigths_path)
        #Eval model:
        fold_logs_path = os.path.join(logs_path, "fold"+str(fold))
        os.makedirs(fold_logs_path)
        eval_nw(modality, os.path.join(root_path_weights,weigths_path), testloader, test_dataset, fold_logs_path)

    #EVAL COMPLETE MODEL




def eval_nw(modality, weights, test_loader, test_dataset, logs_path):
    #Load model:
    if (modality == "saliency" or modality == "landmarks"):
        net = Deep_Emotion_saliency_48x48()
    elif (modality == "original"):
        net = Deep_Emotion_Original_48x48()
    elif (modality == "baseline"):
        net = Deep_Emotion_Baseline_48x48()
    print("Deep Emotion:-", net)
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    #Model Evaluation on test data
    classes = ('Neutral', 'Positive','Negative')

    #Extract predictions:
    if modality == "saliency" or modality == "landmarks":
        preds, labels, idx = eval_landmSaliency(net, test_loader, device)
    else:  # modality == "original" or modality == "baseline":
        preds, labels, idx = eval_original(net, test_loader, device)

    all_labels = labels.cpu().detach().numpy()
    all_preds = preds.cpu().detach().numpy()
    all_idx = idx.cpu().detach().numpy()

    #Extract metrics: accuracy, confussion matrix...
    correct = torch.where(preds == labels, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda()).sum()
    print("Accuracy: ", str(100.0 * (correct / len(all_labels))))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes, os.path.join(logs_path, "cm.png"))
    plt.close()


    df_labels = pd.DataFrame([],columns=[])
    df_labels["preds"] = all_preds
    df_labels["labels"] = all_labels
    df_labels["idx"] = all_idx.astype(int)
    df_labels["path"] = test_dataset.df_file["path"].iloc[all_idx.astype(int)].values

    #save logs:
    os.makedirs(logs_path, exist_ok=True)
    df_labels.to_csv(os.path.join(logs_path, "df_predictions.csv"), sep=";", header=True, index=False)








def eval_landmSaliency(net, test_loader, device):
    all_preds = torch.empty(0)
    all_preds = all_preds.to(device)
    all_labels = torch.empty(0)
    all_labels = all_labels.to(device)
    all_idx = torch.empty(0)
    all_idx = all_idx.to(device)
    with torch.no_grad():
        for data, labels, land, idx in test_loader:
            data, labels, land, idx = data.to(device), labels.to(device), land.to(device), idx.to(device)
            outputs = net(data, land)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            all_preds = torch.cat((all_preds, classs), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_idx = torch.cat((all_idx, idx), dim=0)
            # wrong = torch.where(classs != labels, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
            # acc = 1 - (torch.sum(wrong) / 64)
            # total.append(acc.item())
    return all_preds, all_labels, all_idx


def eval_original(net, test_loader, device):
    all_preds = torch.empty(0)
    all_preds = all_preds.to(device)
    all_labels = torch.empty(0)
    all_labels = all_labels.to(device)
    all_idx = torch.empty(0)
    all_idx = all_idx.to(device)

    with torch.no_grad():
        for data, labels, idx in test_loader:
            data, labels, idx = data.to(device), labels.to(device), idx.to(device)
            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            all_preds = torch.cat((all_preds, classs), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_idx = torch.cat((all_idx, idx), dim=0)
    return all_preds, all_labels, all_idx





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Training file with the dataset files (train.csv and test.csv)')
    parser.add_argument('-r', '--data_root', type=str, required=True,
                        help='Root path with the images')
    parser.add_argument('-l', '--landmark_root', type=str,
                        help='Root path with the landmarks or saliencies')

    parser.add_argument('-bs', '--batch_size', type=int, help='training/validation batch size', default=32)
    parser.add_argument('-logs', '--logs_folder', type=str, help='Path to save logs of evaluation', default='./')
    parser.add_argument('-m','--modality', type=str, help='Choose the architecture of the model (baseline, original, landmarks or saliency)', default="original")
    parser.add_argument('-tl', '--tl_preTrained_weights', type=str, required=True,
                        help='Path to the pre-trained weigths folder with.pt files')

    args = parser.parse_args()

    transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    if args.modality == "saliency":
        test_dataset = Plain_Dataset_saliency(csv_path=args.data, dataroot=args.data_root,
                                              dataroot_land=args.landmark_root, transform=transformation)
    elif args.modality == "landmarks":
        test_dataset = Plain_Dataset_land(csv_path=args.data, dataroot=args.data_root,
                                          dataroot_land=args.landmark_root, transform=transformation)
    else:  # args.modality == "original" or args.modality == "baseline":
        test_dataset = Plain_Dataset(csv_path=args.data, dataroot=args.data_root, transform=transformation)

    eval_5CV(5, args.batch_size, args.tl_preTrained_weights, args.modality, args.logs_folder)







