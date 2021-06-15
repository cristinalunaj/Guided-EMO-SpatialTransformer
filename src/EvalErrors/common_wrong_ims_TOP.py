import os
import numpy as np
import pandas as pd
import src.EvalErrors.processPredictions as processPred

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



if __name__ == '__main__':
    in_imgs = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images"
    in_path = "/mnt/RESOURCES/AFFECTNET/ZZEVAL_ERROR_MODELS/more_problematics_all"
    root_path_imgs = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images"

    complete_df_path = "/mnt/RESOURCES/AFFECTNET/Manually_Annotated_file_lists_ORIGINAL/complete_trainAndVal_NOEXT.csv"
    ascending_order = False
    name_file = "postiveLabelnegativePred.csv" # negativeLabel_positivePred  postiveLabelnegativePred
    n_top = 20
    out_dir = "/mnt/RESOURCES/AFFECTNET/ZZEVAL_ERROR_MODELS/more_problematics_all/EMOTIONS/TOP_POSITIVE_ERRORS"

    complete_df = pd.read_csv(complete_df_path, sep=",", header=0)
    v1 = (1,0)

    df_emotions_errors = pd.DataFrame([])
    for fold in range(0, 5):
        print("FOLD", str(fold))
        df_errors = pd.read_csv(os.path.join(in_path, "fold"+str(fold), name_file), sep=",", header=0)
        df_errors["distance"] = np.sqrt(np.power(df_errors["valence"], 2) + np.power(df_errors["arousal"], 2))
        # angle = []
        # for i, row in df_errors.iterrows():
        #     vector_i = (row["arousal"], row["valence"])
        #     angle_vects = np.math.atan2(np.linalg.det([v1,vector_i]),np.dot(v1,vector_i))
        #     print("to do")
        #     angle.append(np.degrees(angle_vects))
        #
        # df_errors["angle"] = angle
        #df_sorted_by_dist = df_errors.sort_values(by="distance", ascending=ascending_order)
        df_sorted_by_valence = df_errors.sort_values(by="valence", ascending=ascending_order)
        #df_emotions = df_sorted_by_valence.groupby(by="expression")
        df_emotions_errors = df_emotions_errors.append(df_sorted_by_valence)




    df_top_by_emot = pd.DataFrame([])
    df_emotions_errors = df_emotions_errors.sort_values(by="valence", ascending=ascending_order)
    df_emotions_errors = df_emotions_errors.groupby(by="expression")
    for emotion, df_emot in df_emotions_errors:
        df_top = df_emot.head(n=n_top)

        df_top_by_emot = df_top_by_emot.append(df_top)
        #save imgs
        processPred.save_imgs(df_top, out_dir, root_path_imgs, error_type=str(emotion), modality="")


