import os

import pandas as pd
from sklearn.metrics import confusion_matrix
from src.utils.plotcm import plot_confusion_matrix
import matplotlib.pyplot as plt

import cv2
import numpy as np



def analyse_errors(df_erros):
    valence_mean_matrix = np.zeros([3,3])
    valence_std_matrix = np.zeros([3, 3])
    arousal_mean_matrix = np.zeros([3, 3])
    arousal_std_matrix = np.zeros([3, 3])

    # Confussion between Neutral & Positive:
    print("-----> Prediction is Neutral, but label is Postive:")
    df_erros_NeuPos = df_erros.loc[((df_erros["preds"] == 0.0) & (df_erros["labels"] == 1.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal = get_metrics(df_erros_NeuPos)
    valence_mean_matrix[1,0] = mean_val
    valence_std_matrix[1, 0] = std_val
    arousal_mean_matrix[1, 0] = mean_arousal
    arousal_std_matrix[1, 0] = std_arousal
    print("-----> Prediction is Positive, but label is Neutral:")
    df_erros_PosNeu = df_erros.loc[((df_erros["preds"] == 1.0) & (df_erros["labels"] == 0.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal =get_metrics(df_erros_PosNeu)
    valence_mean_matrix[0, 1] = mean_val
    valence_std_matrix[0, 1] = std_val
    arousal_mean_matrix[0, 1] = mean_arousal
    arousal_std_matrix[0, 1] = std_arousal
    print("-----> Prediction is Neutral, but label is Negative:")
    df_erros_NeuNeg = df_erros.loc[((df_erros["preds"] == 0.0) & (df_erros["labels"] == 2.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal =get_metrics(df_erros_NeuNeg)
    valence_mean_matrix[2, 0] = mean_val
    valence_std_matrix[2, 0] = std_val
    arousal_mean_matrix[2, 0] = mean_arousal
    arousal_std_matrix[2, 0] = std_arousal
    print("-----> Prediction is Negative, but label is Neutral:")
    df_erros_NegNeu = df_erros.loc[((df_erros["preds"] == 2.0) & (df_erros["labels"] == 0.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal =get_metrics(df_erros_NegNeu)
    valence_mean_matrix[0, 2] = mean_val
    valence_std_matrix[0, 2] = std_val
    arousal_mean_matrix[0, 2] = mean_arousal
    arousal_std_matrix[0, 2] = std_arousal
    print("--------------------------------------------------------------------------")
    print("-----> Prediction is Negative, but label is Positive:")
    df_erros_PosNeg = df_erros.loc[((df_erros["preds"] == 2.0) & (df_erros["labels"] == 1.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal =get_metrics(df_erros_PosNeg)
    valence_mean_matrix[1, 2] = mean_val
    valence_std_matrix[1, 2] = std_val
    arousal_mean_matrix[1, 2] = mean_arousal
    arousal_std_matrix[1, 2] = std_arousal
    print("-----> Prediction is Positive, but label is Negative:")
    df_erros_PosNeg = df_erros.loc[((df_erros["preds"] == 1.0) & (df_erros["labels"] == 2.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal =get_metrics(df_erros_PosNeg)
    valence_mean_matrix[2, 1] = mean_val
    valence_std_matrix[2, 1] = std_val
    arousal_mean_matrix[2, 1] = mean_arousal
    arousal_std_matrix[2, 1] = std_arousal
    return valence_mean_matrix, valence_std_matrix, arousal_mean_matrix, arousal_std_matrix


def analyse_correct(df_correct, valence_mean_matrix, valence_std_matrix, arousal_mean_matrix, arousal_std_matrix):
    #Positive:
    print("-------------> CORRECT POSITIVES: ")
    df_correct_Pos = df_correct.loc[((df_correct["preds"] == 1.0) & (df_correct["labels"] == 1.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal = get_metrics(df_correct_Pos)
    valence_mean_matrix[1, 1] = mean_val
    valence_std_matrix[1, 1] = std_val
    arousal_mean_matrix[1, 1] = mean_arousal
    arousal_std_matrix[1, 1] = std_arousal

    print("-------------> CORRECT NEUTRAL: ")
    df_correct_Neutral = df_correct.loc[((df_correct["preds"] == 0.0) & (df_correct["labels"] == 0.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal = get_metrics(df_correct_Neutral)
    valence_mean_matrix[0, 0] = mean_val
    valence_std_matrix[0, 0] = std_val
    arousal_mean_matrix[0, 0] = mean_arousal
    arousal_std_matrix[0, 0] = std_arousal
    print("-------------> CORRECT NEGATIVE: ")
    df_correct_Negative = df_correct.loc[((df_correct["preds"] == 2.0) & (df_correct["labels"] == 2.0))]
    n_errors, mean_val, std_val, mean_arousal, std_arousal = get_metrics(df_correct_Negative)
    valence_mean_matrix[2, 2] = mean_val
    valence_std_matrix[2, 2] = std_val
    arousal_mean_matrix[2, 2] = mean_arousal
    arousal_std_matrix[2, 2] = std_arousal

    return valence_mean_matrix, valence_std_matrix, arousal_mean_matrix, arousal_std_matrix



def get_metrics(df, debug=False):
    n_errors, mean_val, std_val = len(df), df["valence"].mean(), df["valence"].std()
    mean_arousal, std_arousal = df["arousal"].mean(), df["arousal"].std()
    if(debug):
        print("# errors: ", len(df))
        print("MEAN VALENCE: ", str(df["valence"].mean()))
        print("std VALENCE: ", str(df["valence"].std()))
        print("MEAN AROUSAL: ", str(df["arousal"].mean()))
        print("std AROUSAL: ", str(df["arousal"].std()))
    return n_errors, mean_val, std_val, mean_arousal, std_arousal

def get_accuracy_per_class(cm, classes):
    acc_classes = []
    for i in range(0,len(cm)):
        acc_class = (cm[i,i]/(cm[i,:].sum()))*100
        print("Accuracy for class ", classes[i], ":", str(acc_class))
        acc_classes.append(acc_class)
    return acc_classes


if __name__ == '__main__':
    complete_df_path = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_file_lists_ORIGINAL/complete_trainAndVal_NOEXT.csv"
    imgs_path = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images" #"/mnt/RESOURCES/AFFECTNET/Manually_Annotated_Images_48x48_grayscale"
    landm_path = "/mnt/RESOURCES/AFFECTNET/LANDMARKS_dlib_MTCNN_PRUEBA"
    classes = ('Neutral', 'Positive', 'Negative')
    labels_default_dataset = ""

    complete_df = pd.read_csv(complete_df_path, sep=",", header=0)

    for fold in range(0, 5):
        root_path_preds = "/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/data/error_analysis/AFFECTNET/1baseline_20210510_225930/fold"+str(fold)+"/baseline"
        path_preds = os.path.join(root_path_preds, "df_predictions.csv")

        df_preds = pd.read_csv(path_preds, sep=";", header=0)
        compl_df_pred = complete_df.loc[complete_df["subDirectory_filePath"].isin(list(df_preds["path"]))]

        df_preds = df_preds.sort_values(by="path")
        df_preds = df_preds.reset_index()
        compl_df_pred = compl_df_pred.sort_values(by="subDirectory_filePath")
        compl_df_pred = compl_df_pred.reset_index()

        df_preds["valence"] = compl_df_pred["valence"]
        df_preds["arousal"] = compl_df_pred["arousal"]

        get_metrics(df_preds)

        #Analyse results
        error_df = df_preds.loc[df_preds["preds"] != df_preds["labels"]]
        valence_mean_matrix, valence_std_matrix, arousal_mean_matrix, arousal_std_matrix = analyse_errors(error_df)
        correct_df = df_preds.loc[df_preds["preds"] == df_preds["labels"]]
        valence_mean_matrix, valence_std_matrix, arousal_mean_matrix, arousal_std_matrix = \
            analyse_correct(correct_df, valence_mean_matrix, valence_std_matrix, arousal_mean_matrix, arousal_std_matrix)

        print("--->>>>>> ACCURACY PER CLASS::")
        cm = confusion_matrix(df_preds["labels"], df_preds["preds"])
        acc_classes = get_accuracy_per_class(cm, classes)
        #Save matrix:
        plot_confusion_matrix(valence_mean_matrix, classes, title="Mean valence", save_path=os.path.join(root_path_preds, "mean_valence.png"),
                              fmt='.2f')
        plot_confusion_matrix(valence_std_matrix, classes, title="Std valence",save_path=os.path.join(root_path_preds, "std_valence.png"),
                              fmt='.2f')
        plot_confusion_matrix(arousal_mean_matrix, classes, title="Mean arousal",save_path=os.path.join(root_path_preds, "mean_arousal.png"),
                              fmt='.2f')
        plot_confusion_matrix(arousal_std_matrix, classes, title="Std arousal",save_path=os.path.join(root_path_preds, "std_arousal.png"),
                              fmt='.2f')
        print("TOTAL Accuracy: ", str((len(df_preds) - len(error_df)) / len(df_preds)))

    # cm = confusion_matrix(df_preds["labels"], df_preds["preds"])
    # plot_confusion_matrix(cm, classes)
    # plt.close()

    # for i in range(0,len(error_df)):
    #     row_img = error_df.iloc[i]
    #     row_original_DS = complete_df.loc[complete_df["subDirectory_filePath"]==row_img["path"]]
    #
    #     img_path = os.path.join(imgs_path, row_img["path"])
    #     img_landm = os.path.join(landm_path, row_img["path"])
    #     print("LABEL: ", classes[int(row_img["labels"])], " -- PRED: ", classes[int(row_img["preds"])])
    #     print("VALENCE: ", str(row_original_DS["valence"].values[0])," -- AROUSAL: ", str(row_original_DS["arousal"].values[0]))
    #     print("-----------------")
    #     #Open imgs:
    #     try:
    #         image1 = cv2.resize(cv2.imread(img_path), (200,200))
    #         image_land = cv2.resize(cv2.imread(img_landm), (200,200))
    #         add = cv2.vconcat([image1, image_land])
    #         # Visualization
    #         #cv2.imshow('image1', image1)
    #         #cv2.imshow('image2', image_land)
    #         #cv2.imshow('image3', image3)
    #         cv2.imshow('Addition', add)
    #         cv2.waitKey(0)
    #     except:continue




