import pandas as pd
import numpy as np
from imutils import face_utils
import cv2
import os
import dlib
from mtcnn.mtcnn import MTCNN
import argparse


def extract_landmarks_img_MTCNN(image_path, in_dataroot, predictor, face_detector, output_dir,
                                output_dir_npy_landm, DS=""):
    if (DS == "FER"):
        out_path = output_dir
        out_path_landm = output_dir_npy_landm
    else:
        out_path = os.path.join(output_dir, image_path.split("/")[0])
        out_path_landm = os.path.join(output_dir_npy_landm, image_path.split("/")[0])
    if(os.path.exists(os.path.join(out_path_landm, image_path.split("/")[-1].split(".")[0]+".npy"))):
        return
    # Load image and convert to gray scale
    image = cv2.imread(os.path.join(in_dataroot, image_path))
    #image = imutils.resize(image, width=width_img_out)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces on image
    detected_faces = face_detector.detect_faces(image)
    #if no face detected, save black image
    if(len(detected_faces)<=0):
        #Black images when no landmarks detected
        background = 255*np.ones(image.shape)
        picture_name = image_path.split("/")[-1]
        if picture_name.split(".")[1] == 'tif':
            picture_name = picture_name.split(".")[0] + '.jpg'
        print("Saving black image for: ", out_path+'/'+picture_name)
        cv2.imwrite(os.path.join(out_path,  picture_name), background) #image_path.split("/")[0],
        return 0

    for n in range(len(detected_faces)):
        x1, y1, width, height = detected_faces[n]['box']
        # Fix a possible bug
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        #Convert to rectangle format:
        rect = dlib.rectangle(x1, y1, x2, y2)
        extract_landmarks_save(gray, image, rect, image_path, predictor,out_path, out_path_landm, DS)
    return 1



def extract_landmarks_img_dlib(image_path, in_dataroot, predictor, face_detector, output_dir, output_dir_npy_landm, DS=""):
    if(DS=="FER"):
        out_path = output_dir
        out_path_landm = output_dir_npy_landm
    else:
        out_path = os.path.join(output_dir, image_path.split("/")[0])
        out_path_landm = os.path.join(output_dir_npy_landm, image_path.split("/")[0])
    if (os.path.exists(os.path.join(out_path_landm, image_path.split("/")[-1].split(".")[0] + ".npy"))):
        return ""
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path_landm, exist_ok=True)
    #Load image and convert to gray scale
    image = cv2.imread(os.path.join(in_dataroot, image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Detect faces on image
    rects = face_detector(gray, 1)
    if(len(rects)<=0):
        return image_path

    for (i, rect) in enumerate(rects):
        extract_landmarks_save(gray, image, rect, image_path, predictor,out_path, out_path_landm, DS)
    return ""





def extract_landmarks_save(gray, image, rect, image_path, predictor, output_dir, output_dir_npy_landm, DS="FER"):
    shape = predictor(gray, rect)
    landmarks_coordinates = face_utils.shape_to_np(shape)
    background = np.zeros(image.shape)
    colors = [(255, 255, 255)] * 8
    complete_img = face_utils.visualize_facial_landmarks(background, landmarks_coordinates, alpha=1, colors=colors)
    picture_name = image_path.split("/")[-1]
    if picture_name.split(".")[1] == 'tif':
        picture_name = picture_name.split(".")[0] + '.jpg'

    cv2.imwrite(os.path.join(output_dir, picture_name), complete_img) #
    #Save landmarks file:
    with open(os.path.join(output_dir_npy_landm, picture_name.split(".")[0]+".npy"), 'wb') as f:
        np.save(f, landmarks_coordinates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-o', '--out_dir_imgs', type=str, required=True, help='Output directory to save generated images with landmarks', default='')

    parser.add_argument('-olandm', '--out_dir_landm', type=str, required=True,
                        help='Output directory to save generated landmarks as .npy files')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to the input images to extract landmarks')

    parser.add_argument('-trainCSV', '--train_csv', type=str, required=True,
                        help='path to the training csv with the path of the images')
    parser.add_argument('-trainCSVSep', '--train_csv_sep', type=str, required=True,
                        help='Separator for the csv [default: ","]', default=",")

    parser.add_argument('-ds', '--dataset_name', type=str, help='Name of the dataset to process ["AffectNet","FER"]', default='AffectNet')
    parser.add_argument('-logs', '--logs', type=str,
                        help='Folder to save logs or extra information generated (as the list of images with no detected landmarks)', default='AffectNet')


    args = parser.parse_args()
    list_black_imgs_out_path = os.path.join(args.logs, args.dataset_name+"lost_imgs_landmarks_extractor_PRUEBA.csv")
    ###############
    # CSV:
    dfTraining = pd.read_csv(args.train_csv, sep=args.train_csv_sep,header=0)
    landmarks_predictor = dlib.shape_predictor("../../data/resources/shape_predictor_68_face_landmarks.dat")
    list_frames = list(dfTraining["path"])

    # 1º Try to detect with dlib
    black_imgs_list = []
    detector = dlib.get_frontal_face_detector()
    for index, row in dfTraining.iterrows():
        image_path = row["path"]
        try:
            black_img_path = extract_landmarks_img_dlib(image_path, args.data, landmarks_predictor, detector, args.out_dir_imgs, args.out_dir_landm,DS=args.dataset_name)
        except:
            black_img_path = image_path
        if(black_img_path!=""):
            black_imgs_list.append(black_img_path)

    # 2º Try to detect with MTCNN previous black imgs
    print("Start MTCNN ... - Lost by dlib: ", str(len(black_imgs_list)))
    detector = MTCNN()
    total_rescued_imgs = 0
    final_imgs_black = []
    for black_img_path in black_imgs_list:
        try:
            result=extract_landmarks_img_MTCNN(black_img_path, args.data, landmarks_predictor, detector, args.out_dir_imgs, args.out_dir_landm,DS=args.dataset_name)
        except:
            result=0
        total_rescued_imgs+=result
        if(result==0):
            final_imgs_black.append(black_img_path)
    print("Total MTCNN rescued imgs: ", str(total_rescued_imgs), " of Lost imgs by dlib", str(len(black_imgs_list)))
    #Save images in a file
    pd.DataFrame(final_imgs_black, columns=["path"]).to_csv(list_black_imgs_out_path, index=False, sep=";", header=True)





