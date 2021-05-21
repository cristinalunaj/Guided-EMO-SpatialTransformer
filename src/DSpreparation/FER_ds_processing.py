import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import StratifiedKFold


def plot_histograms(df, title="",  bins = range(0,9)):
    df_train = df.loc[df["in_set"]=="Training"]
    df_val = df.loc[df["in_set"] == "PublicTest"]
    df_test = df.loc[df["in_set"] == "PrivateTest"]
    #bins = range(0,9)

    plt.hist(df_train["emotion"], bins, alpha=0.3, label='Train', color="b")
    plt.hist(df_val["emotion"], bins, alpha=0.3, label='Public Test', color="y")
    plt.hist(df_test["emotion"], bins, alpha=0.3, label='Private Test', color="r")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()
    print("\n ------------ ")
    print("#TRAINING :", str(len(df_train)))
    print("#VALIDATION :", str(len(df_val)))
    print("#TEST :", str(len(df_test)))
    print("## DISTRIBUTION TRAIN, VAL, TEST:")
    print("## train")
    print(df_train.groupby('emotion').count())
    print("## val")
    print(df_val.groupby('emotion').count())
    print("## test")
    print(df_test.groupby('emotion').count())



def create_polarity_FERds(FER_labes_ds):
    FER_polarity_DS = FER_labes_ds.copy()
    # Remove surprise:
    FER_polarity_DS = FER_polarity_DS.drop(list(FER_polarity_DS.loc[FER_polarity_DS["emotion"]==5].index))
    # New Negative -> label 2 --> previous classes in FER: angry (0), disgust(1), fear (2), sad(4)
    FER_polarity_DS["emotion"].replace([0, 1, 2, 4], 2, inplace=True) #First replace negative values
    # New Positive -> label 1 --> previous classes in FER: happy(3)
    FER_polarity_DS["emotion"].replace([3], [1], inplace=True)  # replace positive values
    # New Neutral -> label 0 --> previous classes in FER: neutral (6)
    FER_polarity_DS["emotion"].replace([6], [0], inplace=True)  # replace neutral values
    #check distribution
    return FER_polarity_DS


def create_folds(FER_DS, folds = 5, seed=2020):
    FER_DS = FER_DS.reset_index(inplace=False)
    FER_DS = FER_DS.drop(["index"], axis=1)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    FER_DS["fold"] = -1
    y = FER_DS["emotion"]
    FER_DS.drop(["emotion"], axis=1)
    X = FER_DS
    fold = 0
    for train_index, test_index in kfold.split(X, y):
        FER_DS.loc[list(test_index), "fold"] = fold
        fold+=1
    return FER_DS


def create_subset_FER_no_landmarks(df_FER_DS, df_lost_imgs, out_FER_DS_withouth_lost_landmarks, out_FER_DS_withouth_lost_landmarks_kfold,
                                   kfold=5, seed=2020):
    df_FER_reduced = df_FER_DS.copy()
    if (os.path.exists(out_FER_DS_withouth_lost_landmarks)):
        df_FER_reduced = pd.read_csv(out_FER_DS_withouth_lost_landmarks, sep=";", header=0)
    else:
        #Remove those images for what we did not find a face
        for i, lost_img in df_lost_imgs.iterrows():
            # look for it and remove
            img_path = lost_img[0]
            df_FER_reduced = df_FER_reduced.drop(df_FER_reduced.loc[df_FER_DS["path"] == img_path].index[0])

        # plot previous and after removing data:
    plot_histograms(df_FER_reduced, title="FER reduced")
    # Save final df:
    #Create 3 polarity classes (+/-/neutral) from emotion classees (sad, angry, happy, neutral...)
    FER_reduced_polatiry = create_polarity_FERds(df_FER_reduced)
    plot_histograms(FER_reduced_polatiry, title="FER complete", bins=range(0, 4))
    #Distribute data in folds:
    FER_reduced_polarity_kfold = create_folds(FER_reduced_polatiry, folds=kfold, seed=seed)
    plot_histograms(FER_reduced_polatiry, title="FER polarity approx. distribution", bins=range(0, 4))
    # Save polarity with folds DS:
    FER_reduced_polarity_kfold.to_csv(out_FER_DS_withouth_lost_landmarks_kfold, sep=";", header=True, index=False)
    print("REAL DISTRIBUTION KFOLD: \n", FER_reduced_polarity_kfold.groupby(['fold', 'emotion']).count())


def create_subset_FER(df_FER_DS, out_path, kfold=5, seed=2020):
    FER_reduced_polatiry = create_polarity_FERds(df_FER_DS)
    plot_histograms(FER_reduced_polatiry, title="FER complete", bins=range(0, 4))
    FER_reduced_polarity_kfold = create_folds(FER_reduced_polatiry, folds=kfold, seed=seed)
    FER_reduced_polarity_kfold.to_csv(out_path, sep=";", header=True, index=False)
    print("REAL DISTRIBUTION KFOLD: \n", FER_reduced_polarity_kfold.groupby(['fold', 'emotion']).count())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-r', '--root_csv', type=str, required=True,
                        help='Root path to the processed DS (/data/datasets_distribution/FER2013/labels_FER2013.csv)')
    parser.add_argument('-kf', '--k_folds', type=int, default=5,
                        help='Number of folds [default: 5]')
    parser.add_argument('-s', '--seed', type=int, default=2020,
                        help='Random seed [default: 2020]')
    parser.add_argument('-o', '--out_dir', type=str,
                        help='Output folder to save generated csv ( with 5cv)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_root_path = "/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/data/datasets_distribution/FER2013"
    out_FER_DS_complete_polarity_kfold =out_root_path+"/labels_FER2013_31885KFOLDimgs.csv"
    seed = 2020
    kfolds = 5



    #df_lost_imgs = pd.read_csv(path_lost_landmarks, header=None)
    df_FER_DS = pd.read_csv(args.root_csv, sep=";", header=0)
    plot_histograms(df_FER_DS, title="FER complete")
    #Create reduced version of FER DS without including images whose faces were not detected
    # & convert to valence/polarity & divide in folds
    # create_subset_FER_no_landmarks(df_FER_DS, df_lost_imgs, out_FER_DS_withouth_lost_landmarks,
    #                                out_FER_DS_withouth_lost_landmarks_kfold, kfold=kfolds, seed=seed)

    #Create reduced version of FER DS & convert to valence/polarity & divide in folds
    create_subset_FER(df_FER_DS, out_FER_DS_complete_polarity_kfold, kfold=kfolds, seed=seed)









