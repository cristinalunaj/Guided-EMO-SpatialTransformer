import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import src.EvalErrors.processPredictions as processPred


def fill_df(df_in_out, df_in):
    for i, row in df_in.iterrows():
        if (row["path"]) in list(df_in_out["path"].values):
            prev_row = df_in_out.loc[df_in_out['path'] == row["path"]]
            df_in_out.loc[df_in_out['path'] == row["path"], "count"] = prev_row["count"].values[0] + 1
        else:
            df2 = pd.DataFrame([[row["path"], 1, row["valence"], row["arousal"], row["expression"]]],
                               columns=["path", "count", "valence", "arousal", "expression"])
            df_in_out = df_in_out.append(df2)
    return df_in_out


if __name__ == '__main__':
    fold = 0
    n_models = 5
    ##AFFECTNET####
    logs_path = "/home/cristinalunaj/PycharmProjects/Guided-EMO-SpatialTransformer/data/error_analysis/AFFECTNET_cd"
    complete_df_path = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_file_lists_ORIGINAL/complete_trainAndVal_NOEXT.csv"
    in_imgs = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images"
    out_path = "/mnt/RESOURCES/AFFECTNET/ZZEVAL_ERROR_MODELS/more_problematics_all"



    avoid_models = ["dilatation2iter", "baseline"]
    dict_positive_label_negative_pred = {}
    dict_negative_label_positive_pred = {}

    complete_df = pd.read_csv(complete_df_path, sep=",", header=0)

    for fold in range(0,5):
        print("FOLD", str(fold))

        df_PosPred_NegLabel = pd.DataFrame([], columns=["path", "count", "valence", "arousal", "expression"])
        df_NegPred_PosLabel = pd.DataFrame([], columns=["path", "count", "valence", "arousal", "expression"])
        for file in os.listdir(logs_path):
            if(file in avoid_models):
                print("Avoid model of: ", file)
                continue
            #if("baseline" in file):continue
            in_logs = os.path.join(logs_path, file)
            df_preds = pd.read_csv(os.path.join(in_logs, "fold"+str(fold), "df_predictions.csv"), sep=";", header=0)

            compl_df_pred = complete_df.loc[complete_df["subDirectory_filePath"].isin(list(df_preds["path"]))]

            df_preds = df_preds.sort_values(by="path")
            df_preds = df_preds.reset_index()
            compl_df_pred = compl_df_pred.sort_values(by="subDirectory_filePath")
            compl_df_pred = compl_df_pred.reset_index()

            df_preds["valence"] = compl_df_pred["valence"]
            df_preds["arousal"] = compl_df_pred["arousal"]
            df_preds["expression"] = compl_df_pred["expression"]

            # Analyse results
            error_df = df_preds.loc[df_preds["preds"] != df_preds["labels"]]
            #Prediction is Negative, but label is Positive - df_erros_PosNeg
            #Prediction is Positive, but label is Negative - df_erros_NegPos
            _, _, _, _, df_erros_PosNeg, df_erros_NegPos = processPred.analyse_errors(error_df)
            # Save imgs of errors
            df_PosPred_NegLabel = fill_df(df_PosPred_NegLabel, df_erros_NegPos)
            df_NegPred_PosLabel = fill_df(df_NegPred_PosLabel, df_erros_PosNeg)



            # for i, row in df_erros_PosNeg.iterrows():
            #     if(row["path"]) in dict_positive_label_negative_pred:
            #         dict_positive_label_negative_pred[row["path"]]+=1
            #     else:
            #         dict_positive_label_negative_pred[row["path"]] = 1
        print("END")
        print("NEGATIVE LABEL - POSITIVE PRED. ")
        #SAVE NEGATIVE LABEL POSITIVE PRED:
        more_problematics_NegPos = df_PosPred_NegLabel.loc[df_PosPred_NegLabel["count"]==n_models-1]
        more_problematics_NegPos = df_PosPred_NegLabel.loc[df_PosPred_NegLabel["count"] == n_models - 1]

        #predominant emotion:
        print(more_problematics_NegPos.groupby('expression').count())

        processPred.save_imgs(more_problematics_NegPos, os.path.join(out_path, "fold" + str(fold), "error"+str(n_models-1)), in_imgs,
                  error_type="NegativeLabel_PositivePred", modality="")
        more_problematics_NegPos.to_csv(os.path.join(out_path, "fold" + str(fold), "negativeLabel_positivePred.csv"))
        #SAVE POSITIVE LABEL NEGATIVE PRED:
        print("POSITIVE LABEL - NEGATIVE PRED. ")
        more_problematics_PosNeg = df_NegPred_PosLabel.loc[df_NegPred_PosLabel["count"]==n_models-1]
        more_problematics_PosNeg = df_NegPred_PosLabel.loc[df_NegPred_PosLabel["count"] == n_models - 1]

        # predominant emotion:
        #predominant emotion:
        print(more_problematics_PosNeg.groupby('expression').count())

        processPred.save_imgs(more_problematics_PosNeg, os.path.join(out_path, "fold" + str(fold), "error"+str(n_models-1)), in_imgs,
                  error_type="PositiveLabel_NegativePred", modality="")
        more_problematics_PosNeg.to_csv(os.path.join(out_path, "fold" + str(fold), "postiveLabelnegativePred.csv"))






