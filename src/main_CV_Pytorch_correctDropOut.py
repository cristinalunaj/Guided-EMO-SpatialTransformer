"""
	File that make a training of one of the implemented architectures in the paper:
	    -Simple-CNN
	    -Original-ST
	    -ST with dilatations (landmarks modality)
	    -ST with landmarks masks (landmarks modality)
	    -ST with soft-landmarks masks (landmarks modality)
	    -ST with saliency maps
	author: Cristina Luna
	date: 06/2021
	Usage:
		(e.g. src/main_CV_Pytorch.py -kf 5 -d ../Guided-EMO-SpatialTransformer/data/datasets_distribution/FER2013/labels_FER2013_31885KFOLDimgs.csv
        -r <FER2013-dir>/IMAGES -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs ../Guided-EMO-SpatialTransformer/data/TL_FER2013_logs -m original
        -tl ../Guided-EMO-SpatialTransformer/data/AFFECTNET_LOGS/2original_20210510_160434/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-241-original.pt
		)
	Options:
		-kf : Integer. Number of folds (default in our experiments: 5)
		-d : Str. Path to the file that contain the dataframe with the data of the routes of images and labels.
		-r : Str. Root path to the input images file
		-imgSize: Integer. Size of the input images to the model [currently only size of 48 is available]
		-e:  Integer. Number of epochs to train the model
		-lr: Float. Learning rate of the model
		-bs:  Integer. (batch_size)Number of samples per batch
		-s: Random seed to replicate experiments
		-logs: Str. Path to save logs and models
		-tl: Str. Path to pre-trained weights if we want to apply transfer-learning, if not: None
"""
from __future__ import print_function
import argparse

import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data_loaders.data_loaders import Plain_Dataset
from src.data_loaders.data_loader_land import Plain_Dataset_land
from src.data_loaders.data_loader_saliency import Plain_Dataset_saliency

from src.architectures.deep_emotion_saliency import Deep_Emotion_Saliency as Deep_Emotion_saliency_48x48
from src.architectures.deep_emotion_original import Deep_Emotion_Original as Deep_Emotion_Original_48x48
from src.architectures.deep_emotion_baseline import Deep_Emotion_Baseline as Deep_Emotion_Baseline_48x48

from torch.utils.tensorboard import SummaryWriter

from src.utils.args_utils import str2bool
#REDUCE RANDOMNESS:
import random
import numpy as np
from datetime import datetime





def seed_torch(seed=2020):
    """
    Fix random seeds to allow replications
        :param seed: Seed to introduce for genereting the random numbers
    """
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
        :param epochs: Integer. Number of epochs to train the model
        :param k_folds: Integer. Number of folds (default in our experiments: 5)
        :param train_dataset: Plain_Dataset class. DataLoader (check data_loaders.py) lo load images of the training set
        :param test_dataset: Plain_Dataset class.DataLoader (check data_loaders.py) lo load images of the test set
        :param device: Str. Device to use (cpu or gpu). It was detected previously
        :param img_size: Integer. Size of the input images to the model [currently only size of 48 is available]
        :param modality: Str. Model to use for training: [Options: saliency/landmarks/original/baseline]
        :param class_weigths: Boolean. If True, calculate the weigths associated to the training file to balance the training.
        :param batch_size: Integer. Number of samples per batch
        :param lr: Float. Learning rate of the model
        :param num_workers: Integer. Number of workers to load images in parallel during training.
        :param num_epochs_stop: Integer. Early Stopping parameter. Number of patience epochs to stop training.
        :param preTrainedWeigths: Str. Path to pre-trained weights if we want to apply transfer-learning, if not: None
        :param train_complete_model_flag: Boolean. True if we want to train the final model with all the data of the dataset.
        :param save_path: Str. Path to save logs and models
    '''

    print("SAVING DATA IN: ", (save_path))
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

        if(class_weigths):
            # Obtain classes for training with unbalanced DS
            class_weigths_values = check_balance_in_data(train_dataset.df_file.loc[test_dataset.df_file["fold"] != fold])
        else:
            class_weigths_values = None

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
        criterion = nn.CrossEntropyLoss(weight=class_weigths_values)
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
            net.trainingState = True

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
            net.trainingState=False
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
                last_top_acc = min_val_acc#100.0 * (val_correct / len(test_ids))
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
    print(f'Average convergence epochs: {epochs2stop / len(results.items())} ')

    # RUN & SAVE LAST MODEL TRAINING
    if (train_complete_model_flag):
        print("TRAINING LAST MODEL ...")
        if (class_weigths):
            # Obtain classes for training with unbalanced DS
            class_weigths_values = check_balance_in_data(train_dataset.df_file)
        else:
            class_weigths_values = None
        writer = SummaryWriter(log_dir=os.path.join(save_path, "logs", "COMPLETE"))
        train_complete_model(epochs, train_dataset, device, writer, img_size=img_size, class_weigths=class_weigths_values,
                             batch_size=batch_size, lr=lr, save_path=save_path, modality=modality, num_workers=num_workers,
                             num_epochs_stop=num_epochs_stop)
        writer.flush()
        writer.close()



def train_model_original(net, train_loader, optmizer, criterion, device):
    """
    Train one epoch of the original or baseline modalities . Original modality is the Baseline-ST and baseline modality is the Simple-CNN.
        :param net: Network to train
        :param train_loader: Training loader
        :param optmizer: optimizer
        :param criterion: Loss function (defined before)
        :param device: cpu or gpu
        :return:
            train_loss: Loss of the network for that epoch
            train_correct: Number of correct samples in that epoch
    """
    train_loss = 0
    train_correct = 0
    for data, labels, _ in train_loader:
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
    """
        Train one epoch of the saliency-based or landmarks-based modalities.
            :param net: Network to train
            :param train_loader: Training loader
            :param optmizer: optimizer
            :param criterion: Loss function (defined before)
            :param device: cpu or gpu
            :return:
                train_loss: Loss of the network for that epoch
                train_correct: Number of correct samples in that epoch
        """
    train_loss = 0
    train_correct = 0
    for data, labels, saliency, _ in train_loader:
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
    """
        Eval the samples of the validation/test set for saliency-based or landmarks-based modalities.
            :param net: Network to train
            :param testloader: Test/validation loader
            :param criterion: Loss function (defined before)
            :param device: cpu or gpu
            :return:
                validation_loss: Loss of the network for the samples
                val_correct: Number of correct samples
        """
    val_correct, validation_loss = 0, 0
    with torch.no_grad():
        for data, labels, land,_ in testloader:
            data, labels, land = data.to(device, non_blocking=True), labels.to(device, non_blocking=True), land.to(device, non_blocking=True)
            outputs = net(data, land)
            validation_loss += criterion(outputs, labels).item()
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            val_correct += (classs == labels).sum().item()
    return validation_loss, val_correct

def eval_model_original(net, testloader, criterion, device):
    """
        Eval the samples of the validation/test set for original or baseline modalities . Original modality is the Baseline-ST and baseline modality is the Simple-CNN.
            :param net: Network to train
            :param testloader: Test/validation loader
            :param criterion: Loss function (defined before)
            :param device: cpu or gpu
            :return:
                validation_loss: Loss of the network for the samples
                val_correct: Number of correct samples
        """
    val_correct, validation_loss = 0, 0
    with torch.no_grad():
        for data, labels, _ in testloader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = net(data)
            validation_loss += criterion(outputs, labels).item()
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            val_correct += (classs == labels).sum().item()
    return validation_loss, val_correct


def train_complete_model(epochs,train_dataset,device, writer,img_size, class_weigths, batch_size, lr,
          save_path, modality, num_workers, num_epochs_stop):
    """
    Train the complete model with all the samples of the dataset
        :param epochs: Integer. Number of epochs to train the model
        :param train_dataset: Plain_Dataset class. DataLoader (check data_loaders.py) lo load images of the training set
        :param device: Str. Device to use (cpu or gpu). It was detected previously
        :param writer: SummaryWriter class. To dave training information and models.
        :param img_size: Integer. Size of the input images to the model [currently only size of 48 is available]
        :param modality: Str. Model to use for training: [Options: saliency/landmarks/original/baseline]
        :param class_weigths: Dict. Weights used to multiply the loss function depending on th enumber of samples of each class
        :param batch_size: Integer. Number of samples per batch
        :param lr: Float. Learning rate of the model
        :param num_workers: Integer. Number of workers to load images in parallel during training.
        :param num_epochs_stop: Integer. Early Stopping parameter. Number of patience epochs to stop training.
        :param save_path: Str. Path to save logs and models
    """
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

    epochs_no_improve = 0
    min_val_acc = 0
    last_top_acc = 0
    last_top_acc_epoch = 0
    epochs2converge = 0
    results = 0
    print("------------------ START TRAINING of fold ---------------------")
    for e in range(epochs):
        print('\nEpoch {} / {} - FINAL MODEL'.format(e + 1, epochs))
        # Train the model  #
        net.train()
        net.trainingState = True
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
        # EARLY STOPPING:
        if train_acc > min_val_acc:
            # Save the model
            # Save BEST weigths to recover them posteriorly
            torch.save(net.state_dict(),
                       os.path.join(save_path, "trained_models",
                                    'TMP-deep_emotion-{}-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr,
                                                                                      "COMPLETE",
                                                                                      img_size, e, modality)))
            epochs_no_improve = 0
            min_val_acc = train_acc
            last_top_acc = 100.0 *(train_correct.double() / len(train_dataset))
            last_top_acc_epoch = e
            epochs2converge = e

        else:
            epochs_no_improve += 1

        if e > (num_epochs_stop - 1) and epochs_no_improve == num_epochs_stop:
            print('Early stopping IN EPOCH: !', str(e), " - Best weigths saved in TMP-deep_emotion....")
            # EARLY STOPPING
            results = last_top_acc
            epochs2converge = last_top_acc_epoch
            break

    print("EPOCHS TO CONVERGE: ", str(epochs2converge), ", ACC: ", str(results))
    # save final model
    torch.save(net.state_dict(),
               os.path.join(save_path, "trained_models",
                            'deep_emotion-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr, "COMPLETE", img_size,
                                                                       modality)))
    print("===================================Training Finished===================================")


def check_balance_in_data(df_data):
    """
    Extract the weights associated to the classes in unbalanced tasks.
        :param df_data: Dataset with all the data to extract the weigths. (Normally training data)
        :return: Dict with the weights of each class
    """
    #df_data = pd.read_csv(traincsv_file, header=0)
    #PLOT LABELS TO SEE DISTRIBUTION OF DATA
    labels_complete = df_data["emotion"]
    classes = labels_complete.unique()
    #import matplotlib.pyplot as plt
    #plt.hist(labels_complete)
    #plt.xlabel("Classes")
    #plt.ylabel("Number of samples per class")
    #plt.xticks(classes)
    #plt.show()
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
    parser.add_argument("--train",
                        type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Train the complete model after cross-validation. ")
    parser.add_argument('-tl', '--tl_preTrained_weights', type=str, required=False, default=None,
                        help='Path to the pre-trained weigths (.pt file)')

    args = parser.parse_args()

    print("PROCESSING MODALITY: ", args.modality)
    print("WEIGHTS: ", args.tl_preTrained_weights)

    #Prepare environment:
    os.environ["PYTHONWARNINGS"] = "ignore"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    seed_torch(seed=args.seed)

    # Create out folder for logs and models:
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")


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
    Train(args.epochs, args.kfolds,train_dataset, test_dataset, device, class_weigths=True,
          batch_size=args.batch_size, img_size=args.img_size, lr=args.learning_rate, num_workers=6,
          modality=args.modality, num_epochs_stop=30 , train_complete_model_flag=args.train, preTrainedWeigths=args.tl_preTrained_weights,
          save_path=os.path.join(args.logs_folder, current_time))


