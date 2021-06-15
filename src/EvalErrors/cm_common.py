import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from src.utils.plotcm import plot_confusion_matrix
import src.EvalErrors.processPredictions as processPred

if __name__ == '__main__':
    #AFFECTNET####################
    model = "soft_landmarks"
    logs_path = "/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/data/error_analysis/AFFECTNET_cd/"+model
    out_path = "/mnt/RESOURCES/AFFECTNET/ZZEVAL_ERROR_MODELS/"+model
    in_imgs = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images"


    complete_df_path = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_file_lists_ORIGINAL/complete_trainAndVal_NOEXT.csv"
    no_detected_facial_detector = "/mnt/RESOURCES/AFFECTNET/AffectNetlost_imgs_landmarks_extractor_PRUEBA.csv"
    sep_complete_df = ","
    # FER-2013 ####################
    # model = "softLandmarks"
    # logs_path = "/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/data/error_analysis/FER2013_TL_cd/" + model
    # out_path = "/mnt/ESITUR2/DATASETS/FER2013/fer2013/ZZEVAL_ERROR_MODELS/" + model
    # in_imgs = "/mnt/ESITUR2/DATASETS/FER2013/fer2013/IMAGES"
    #
    # complete_df_path = "/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/data/datasets_distribution/FER2013/labels_FER2013_31885KFOLDimgs.csv"
    # no_detected_facial_detector = "/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/data/datasets_distribution/FER2013/lost_imgs_landmarks_extractor.csv"
    # sep_complete_df = ";"

    ###############################
    classes = ('Neutral', 'Positive','Negative')
    cm_matrix = np.zeros((3,3))
    no_detected_df = pd.read_csv(no_detected_facial_detector, sep=",", header=0)
    list_no_detected_faces = list(no_detected_df["path"].values)
    complete_df = pd.read_csv(complete_df_path, sep=sep_complete_df, header=0)
    errors_total_matches = 0


    for fold in range(0,5):
        logs_path_fold = os.path.join(logs_path, "fold"+str(fold), "df_predictions.csv")
        fold_df = pd.read_csv(logs_path_fold, sep=";", header=0)
        paths_wrong = list(fold_df.loc[fold_df["preds"]!=fold_df["labels"], "path"].values)
        match_errors = set(paths_wrong).intersection(set(list_no_detected_faces))
        errors_total_matches+=len(match_errors)
        print("f", str(match_errors))
        # save imgs
        df_imgs = pd.DataFrame(list(match_errors), columns=["path"])
        processPred.save_imgs(df_imgs,
                              os.path.join(out_path, "fold" + str(fold)), in_imgs,
                              error_type="Face_detector_TrueErrors", modality="")

    print("ERRORS BECAUSE NON-DETECTIONS: ", str(errors_total_matches), "/", str(len(list_no_detected_faces)))

    #Valence per emotion:
    # complete_df.groupby(by="expression")
    # print("MEAN INFO: \n", complete_df.groupby(by="expression")["valence"].mean())
    # print("STD INFO: \n", complete_df.groupby(by="expression")["valence"].std())



    #GET AVG. CONFUSION MATRIX FOR 5-CV
    for fold in range(0,5):
        logs_path_fold = os.path.join(logs_path, "fold"+str(fold), "df_predictions.csv")
        fold_df = pd.read_csv(logs_path_fold, sep=";", header=0)
        cm = confusion_matrix(fold_df["labels"], fold_df["preds"])
        cm_matrix+=cm
    cm_matrix/=5
    cm_matrix = cm_matrix.astype(int)
    #save average confussion matrix
    plot_confusion_matrix(cm_matrix, classes, os.path.join(logs_path, "avg_cm.png"))

