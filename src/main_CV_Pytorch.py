from __future__ import print_function
import argparse

import pandas as pd
import sys,os
sys.path.append("../..")
sys.path.append("/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer")
sys.path.append("/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/src")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_loaders.data_loaders import Plain_Dataset
from data_loaders.data_loader_land import Plain_Dataset_land
from data_loaders.data_loader_saliency import Plain_Dataset_saliency

from architectures.deep_emotion_saliency import Deep_Emotion_Saliency as Deep_Emotion_saliency_48x48
from architectures.deep_emotion_original import Deep_Emotion_Original as Deep_Emotion_Original_48x48
from architectures.deep_emotion_baseline import Deep_Emotion_Baseline as Deep_Emotion_Baseline_48x48

from torch.utils.tensorboard import SummaryWriter
#REDUCE RANDOMNESS:
import random
import numpy as np
from datetime import datetime




def seed_torch(seed=2020):
    random.seed(seed)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def Train(epochs,k_folds,train_dataset,test_dataset,device,img_size, modality,class_weigths, batch_size, lr,num_workers,num_epochs_stop,
          preTrainedWeigths, train_complete_model_flag, save_path):

    '''
    Training Loop
    '''

    print("SAVING DATA IN: ", (save_path))
     #kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}
    epochs2converge={}
    print("===================================Start Training===================================")
    for fold in range(0, k_folds):
        train_ids = np.array(list(train_dataset.df_file.loc[test_dataset.df_file["fold"] != fold].index))
        test_ids = np.array(list(test_dataset.df_file.loc[test_dataset.df_file["fold"] == fold].index))
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        print("N_imgs_training: ", str(len(train_ids)))
        print("N_imgs_test: ", str(len(test_ids)))

        # Print
        # CREATE WRITER PER FOLD:
        writer = SummaryWriter(log_dir=os.path.join(save_path, "logs", "fold_" + str(fold)))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler,
                                  num_workers=num_workers, pin_memory=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=num_workers,
                                pin_memory=True)


        if (modality == "saliency" or modality == "landmarks"):
            net = Deep_Emotion_saliency_48x48()
        elif (modality == "original"):
            net = Deep_Emotion_Original_48x48()
        elif (modality == "baseline"):
            net = Deep_Emotion_Baseline_48x48()

        # START LEARNING WITH PRE-TRAINED WEIGHTS FROM OTHER DS (OR NOT) - TRANSFER LEARNING
        if (preTrainedWeigths != None):
            print("Transfer learning of the model...")
            net.load_state_dict(torch.load(preTrainedWeigths))

        net.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weigths)
        optmizer = optim.Adam(net.parameters(), lr=lr)
        # Create output directory of nws:
        os.makedirs(os.path.join(save_path, "trained_models"), exist_ok=True)
        # Early Stopping parameters:
        epochs_no_improve = 0
        min_val_acc = 0
        last_top_acc = 0
        last_top_acc_epoch = 0

        print("------------------ START TRAINING of fold ---------------------")
        for e in range(epochs):
            print('\nEpoch {} / {} \nFold number {} / {}'.format(e + 1, epochs, fold + 1, k_folds))
            # Train the model  #
            net.train()

            if modality == "saliency" or modality == "landmarks":
                train_loss, train_correct = train_model_saliency(net, train_loader, optmizer, criterion, device)
            else:# modality == "original" or modality == "baseline":
                train_loss, train_correct = train_model_original(net, train_loader, optmizer, criterion, device)

            print(">>>>>>>>>>>> ON EPOCH: ", str(e))
            print(">> training : ")
            train_loss = (train_loss / len(train_ids))
            writer.add_scalar("Loss/train", train_loss, e)
            train_acc = train_correct.double() / len(train_ids) * 100
            writer.add_scalar("Accuracy/train", train_acc, e)

            print('TRAINING: Fold: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(fold, e, train_loss,train_acc))

            #evaluate the fold#
            print(">> eval : ")
            net.eval()
            if modality == "saliency" or modality == "landmarks":
                validation_loss, val_correct = eval_model_saliencyORlandmarks(net, testloader, criterion, device)
            else: #modality == "original" or modality == "baseline":
                validation_loss, val_correct = eval_model_original(net, testloader, criterion, device)

            validation_loss = validation_loss / len(test_ids)
            results[fold] = 100.0 * (val_correct / len(test_ids))
            epochs2converge[fold] = e
            # Print accuracy
            print('VALIDATION: Fold: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(fold, e, validation_loss, results[fold]))
            print('--------------------------------')

            writer.add_scalar("Loss/val", validation_loss, e)
            writer.add_scalar("Accuracy/val", results[fold], e)
            # Send data at the end of the epoch
            writer.flush()

            # EARLY STOPPING:
            if results[fold] > min_val_acc:
                # Save the model
                # Save BEST weigths to recover them posteriorly
                torch.save(net.state_dict(),
                           os.path.join(save_path, "trained_models",
                                        'TMP-deep_emotion-{}-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr,
                                                                                                fold,
                                                                                                img_size, e, modality)))
                epochs_no_improve = 0
                min_val_acc = results[fold]
                last_top_acc = 100.0 * (val_correct / len(test_ids))
                last_top_acc_epoch = e
                epochs2converge[fold] = e

            else:
                epochs_no_improve += 1

            if e > (num_epochs_stop - 1) and epochs_no_improve == num_epochs_stop:
                print('Early stopping IN EPOCH: !', str(e), " - Best weigths saved in TMP-deep_emotion....")
                # EARLY STOPPING
                results[fold] = last_top_acc
                epochs2converge[fold] = last_top_acc_epoch
                break

        # save LAST MODEL
        torch.save(net.state_dict(),
                   os.path.join(save_path, "trained_models",
                                'deep_emotion-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr,
                                                                                         fold, img_size,modality)))
        print("===================================Training Finished===================================")
        # Close tensorboard writer
        writer.flush()
        writer.close()

        # RECOVER BEST MODEL BASED ON VALIDATION:
        top_model_weights = os.path.join(save_path, "trained_models",
                                         'TMP-deep_emotion-{}-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size,
                                                                                                 lr, fold,
                                                                                                 img_size,
                                                                                                 last_top_acc_epoch,
                                                                                           modality))
        print(">>> Loading TOP nw for test eval: ", top_model_weights)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    epochs2stop = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
        epochs2stop+=epochs2converge[key]
    print(f'Average: {sum / len(results.items())} %')
    print(f'Average convergence epochs: {epochs2stop / len(results.items())} %')

    # RUN & SAVE LAST MODEL TRAINING
    if (train_complete_model_flag):
        print("TRAINING LAST MODEL ...")
        writer = SummaryWriter(log_dir=os.path.join(save_path, "logs", "COMPLETE"))
        train_complete_model(max(epochs2converge.values())+10, train_dataset, device, writer, img_size=img_size, class_weigths=class_weigths,
                             batch_size=batch_size, lr=lr, save_path=save_path, modality=modality, num_workers=num_workers)
        writer.flush()
        writer.close()



def train_model_original(net, train_loader, optmizer, criterion, device):
    train_loss = 0
    train_correct = 0
    for data, labels in train_loader:
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optmizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

        loss.backward()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

        #Update model's weigths
        optmizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    return train_loss, train_correct

def train_model_saliency(net, train_loader, optmizer, criterion, device):
    train_loss = 0
    train_correct = 0
    for data, labels, saliency in train_loader:
        data, labels, saliency = data.to(device, non_blocking=True), labels.to(device, non_blocking=True), saliency.to(device, non_blocking=True)
        optmizer.zero_grad()
        outputs = net(data,saliency)
        loss = criterion(outputs, labels)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

        loss.backward()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

        #Update model's weigths
        optmizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    return train_loss, train_correct

def eval_model_saliencyORlandmarks(net, testloader, criterion, device):
    val_correct, validation_loss = 0, 0
    with torch.no_grad():
        for data, labels, land in testloader:
            data, labels, land = data.to(device, non_blocking=True), labels.to(device, non_blocking=True), land.to(device, non_blocking=True)
            outputs = net(data, land)
            validation_loss += criterion(outputs, labels).item()
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            val_correct += (classs == labels).sum().item()
    return validation_loss, val_correct

def eval_model_original(net, testloader, criterion, device):
    val_correct, validation_loss = 0, 0
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = net(data)
            validation_loss += criterion(outputs, labels).item()
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            val_correct += (classs == labels).sum().item()
    return validation_loss, val_correct


def train_complete_model(epochs,train_dataset,device, writer,img_size, class_weigths, batch_size, lr,
          save_path, modality, num_workers):

    train_loader= DataLoader(train_dataset,batch_size=batch_size,shuffle = True,num_workers=num_workers, pin_memory=True)

    if img_size == 48 and (modality == "saliency" or modality == "landmarks"):
        net = Deep_Emotion_saliency_48x48()
    elif img_size == 48 and modality == "original":
        net = Deep_Emotion_Original_48x48()
    elif img_size == 48 and modality == "baseline":
        net = Deep_Emotion_Baseline_48x48()

    net.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weigths)
    optmizer = optim.Adam(net.parameters(), lr=lr)
    print("------------------ START TRAINING of fold ---------------------")
    for e in range(epochs):
        print('\nEpoch {} / {} - FINAL MODEL'.format(e + 1, epochs))
        # Train the model  #
        net.train()
        # Train 1 epoch
        if modality == "saliency" or modality == "landmarks":
            train_loss, train_correct = train_model_saliency(net, train_loader, optmizer,
                                                             criterion, device)
        else: # modality == "original" or modality == "baseline":
            train_loss, train_correct = train_model_original(net, train_loader, optmizer,
                                                             criterion, device)

        # Print validation accuracy & save
        print(">>>>>>>>>>>> ON EPOCH: ", str(e))
        print(">> training : ")
        train_loss = (train_loss / len(train_dataset))
        print("TRAIN LOSS: ", str(train_loss))
        writer.add_scalar("Loss/train", train_loss, e)
        train_acc = 100*(train_correct.double() / len(train_dataset))
        print("Train Accuracy: ", str(train_acc))
        writer.add_scalar("Accuracy/train", train_acc, e)


    # save final model
    torch.save(net.state_dict(),
               os.path.join(save_path, "trained_models",
                            'deep_emotion-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr, "COMPLETE", img_size,
                                                                       modality)))
    print("===================================Training Finished===================================")


def check_balance_in_data(traincsv_file):
    df_data = pd.read_csv(traincsv_file, header=0)
    #PLOT LABELS TO SEE DISTRIBUTION OF DATA
    labels_complete = df_data["emotion"]
    classes = labels_complete.unique()
    import matplotlib.pyplot as plt
    plt.hist(labels_complete)
    plt.xlabel("Classes")
    plt.ylabel("Number of samples per class")
    plt.xticks(classes)
    plt.show()
    #EXTRACT WEIGHTS TO COMPENSATE UNBALANDED DATASETS
    dict_classes_weigths = {}
    for cls in classes:
        n_clss = len(df_data.loc[df_data["emotion"] == cls])
        dict_classes_weigths[cls] = n_clss
    min_v = min(list(dict_classes_weigths.values()))
    for cls in classes:
        dict_classes_weigths[cls] = min_v/dict_classes_weigths[cls]
    #order dict by key:
    sorted_dict = dict(sorted(dict_classes_weigths.items()))
    return torch.FloatTensor(list(sorted_dict.values())).cuda()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-kf', '--kfolds', type=int, help='Number of folds if CV', default=5)
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Training file with the dataset files (train.csv and test.csv)')
    parser.add_argument('-r', '--data_root', type=str, required=True,
                        help='Root path with the images')
    parser.add_argument('-l', '--landmark_root', type=str,
                        help='Root path with the landmarks or saliencies')
    parser.add_argument('-imgSize', '--img_size', type=int,
                        help='Type of model to use, with input images of 48x48. Options:[48]',
                        default=48)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, help='value of learning rate', default=0.001)
    parser.add_argument('-bs', '--batch_size', type=int, help='training/validation batch size', default=32)
    parser.add_argument('-s', '--seed', type=int, help='Seed to feed random generators', default=2020)
    parser.add_argument('-logs', '--logs_folder', type=str, help='Path to save logs of training', default='./')
    parser.add_argument('-m','--modality', type=str, help='Choose the architecture of the model (baseline, original, landmarks or saliency)', default="original")
    parser.add_argument('-t', '--train', type=bool, help='Train the complete model after cross-validation', default=False)

    args = parser.parse_args()

    #Prepare environment:
    os.environ["PYTHONWARNINGS"] = "ignore"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    seed_torch(seed=args.seed)

    # Create out folder for logs and models:
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")

    #Obtain classes for training with unbalanced DS
    class_weigths = check_balance_in_data(args.data)

    #Convert input images to expected nw size (48x48) or (100x100) and normalize values
    transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    # Create DS generator
    train_dataset = None
    if args.modality == "saliency":
        train_dataset= Plain_Dataset_saliency(csv_path=args.data, dataroot=args.data_root,
                                              dataroot_land=args.landmark_root, transform = transformation)
        test_dataset = Plain_Dataset_saliency(csv_path=args.data, dataroot=args.data_root,
                                              dataroot_land=args.landmark_root, transform = transformation)
    elif args.modality == "landmarks":
        train_dataset = Plain_Dataset_land(csv_path=args.data, dataroot=args.data_root,
                                           dataroot_land=args.landmark_root, transform=transformation)
        test_dataset = Plain_Dataset_land(csv_path=args.data, dataroot=args.data_root,
                                          dataroot_land=args.landmark_root, transform=transformation)

    else:# args.modality == "original" or args.modality == "baseline":
        train_dataset = Plain_Dataset(csv_path=args.data, dataroot=args.data_root, transform=transformation)
        test_dataset = Plain_Dataset(csv_path=args.data, dataroot=args.data_root, transform=transformation)


    #Train nw
    Train(args.epochs, args.kfolds,train_dataset, test_dataset, device, class_weigths=class_weigths,
          batch_size=args.batch_size, img_size=args.img_size, lr=args.learning_rate, num_workers=6,
          modality=args.modality, num_epochs_stop=30 , train_complete_model_flag=args.train, preTrainedWeigths=None,
          save_path=os.path.join(args.logs_folder, current_time))

