import os.path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def convert2polarity(df):
    neutral_imgs = df[(df["valence"] <= 0.2) & (df["valence"] >= -0.2)]
    neutral_imgs["emotion"] = 0
    positive_imgs = df[(df["valence"] > 0.2) & (df["valence"] <= 1)]
    positive_imgs["emotion"] = 1
    negative_imgs = df[(df["valence"] < -0.2) & (df["valence"] >= -1)]
    negative_imgs["emotion"] = 2
    complete_df = pd.concat([positive_imgs, neutral_imgs, negative_imgs])
    polatiry_df = pd.DataFrame()
    complete_df = complete_df.reset_index()
    polatiry_df["path"] = complete_df["subDirectory_filePath"]
    polatiry_df["emotion"] = complete_df["emotion"]
    return polatiry_df



def remove_expression(df, expression_number=5):
    print("to do")
    df.drop(list(df.loc[df["expression"]==expression_number].index))

def create_folds(in_df, folds = 5, seed=2020):
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    in_df["fold"] = -1
    y = in_df["emotion"]
    in_df.drop(["emotion"], axis=1)
    X = in_df
    fold = 0
    for train_index, test_index in kfold.split(X, y):
        in_df.loc[list(test_index), "fold"] = fold
        fold+=1
    return in_df


def check_imgs_exist(df_file, path_images):
    for i, row in df_file.iterrows():
        if(not os.path.isfile(os.path.join(path_images, row["path"]))):
            print("IMAGE DOES NOT EXIST:", str(row["path"]))

def convertExtension(df_file, new_extension=".png"):
    df_file["path"] = df_file["path"].replace("\.[^.]*$", new_extension, regex=True)
    return df_file



if __name__ == '__main__':
    dict_AffectNet_expressions = {0:"Neutral",
                                  1:"Happy",
                                  2:"Sad",
                                  3:"Surprise",
                                  4:"Fear",
                                  5:"Disgust",
                                  6:"Anger",
                                  7:"Contempt",
                                  8:"None",
                                  9:"Uncertain",
                                  10:"NonFace"}
    #### Input parameters:
    train_csv = '/mnt/RESOURCES/AFFECTNET/Manually_Annotated_file_lists_ORIGINAL/training.csv'
    validation_csv = '/mnt/RESOURCES/AFFECTNET/Manually_Annotated_file_lists_ORIGINAL/validation.csv'
    kfolds = 5
    ### Out parameters:
    out_polarity_complete_csv = '../../data/datasets_distribution/AffectNet/polarity_complete.csv'
    out_polarity_complete_csv_5folds = '../../data/datasets_distribution/AffectNet/polarity_complete_5folds.csv'


    ### CODE
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(validation_csv)
    df_total = pd.concat([df_train, df_val])

    #Images per class/emotion:
    #Number of samples per emotion:
    print(df_total.groupby("expression").count())
    print("Total images in DS: ", str(len(df_total)))
    polarity_df = convert2polarity(df_total)
    #save polarity df complete:
    polarity_df.to_csv(out_polarity_complete_csv, sep=";", index=False, header=True)

    # Distribute emotions in folds:
    polarity_df_folds = create_folds(polarity_df, folds=kfolds, seed=2020)
    # save polarity df complete - folds:
    polarity_df_folds.to_csv(out_polarity_complete_csv_5folds, sep=";", index=False, header=True)
    print("Distribution of images per fold:")
    print(polarity_df_folds.groupby(['fold', 'emotion']).count())
    
    #Remove duplicated images if exist:
    #polarity_df = polarity_df.drop_duplicates(subset="path")
    #df_polarity_folds_png = convertExtension(polarity_df, new_extension=".png")
    #check_imgs_exist(polarity_df_folds, "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images")



