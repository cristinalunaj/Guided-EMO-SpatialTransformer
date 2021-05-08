import numpy as np
import cv2
import os
import time, multiprocessing
from functools import partial
import collections
import argparse

FACIAL_LANDMARKS_68_IDXS = collections.OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


def create_img_part_face(image, shape, name_face2rm, alpha=1):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    colors = (255,255,255)

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        if(name in name_face2rm):continue
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]

        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors, 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors, -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output




def obtain_landmarks_parallel(list_subfolder, root_folder_original_imgs, root_path_landm_npy, out_dir, part_of_face2rm):
    """
    Extract frames in a parallel way using ffmpeg or opencv funcions
    """
    start_time = time.time()
    pool = multiprocessing.Pool()  # processes = 7

    landmarks_rm_part = partial(create_removed_part_img,  # get_npy_with_previous_clip_AVEC2019
                               root_folder=root_folder_original_imgs,
                               root_path_landm_npy = root_path_landm_npy,
                                out_dir=out_dir,
                                part_of_face=part_of_face2rm)

    pool.map(landmarks_rm_part, list_subfolder)

    pool.close()
    pool.join()
    final_time = (time.time() - start_time)
    print("--- %s Data preparation TIME IN min ---" % (final_time / 60))



def remove_part_of_face(img, part_of_face, landmarks_coordinates):
    img_face = create_img_part_face(img, landmarks_coordinates, part_of_face)
    return img_face


def create_removed_part_img(subfolder, root_folder, root_path_landm_npy, out_dir, part_of_face):
    path_imgs = os.path.join(root_folder, subfolder)
    out_folder = os.path.join(out_dir, subfolder)
    os.makedirs(out_folder, exist_ok=True)
    for img in os.listdir(path_imgs):
        #Load image:
        image = cv2.imread(os.path.join(path_imgs, img))
        #Load npy:
        path_landm = os.path.join(root_path_landm_npy, subfolder, img.split(".")[0]+".npy")
        if(os.path.exists(path_landm)):
            #Load landmarks:
            black_img = np.zeros(np.shape(image))
            with open(os.path.join(root_path_landm_npy, subfolder, img.split(".")[0] + ".npy"), 'rb') as f:
                landmarks_coordinates = np.load(f)
            landm_face_img = remove_part_of_face(black_img, part_of_face, landmarks_coordinates)
            save_img(landm_face_img, out_folder, img)
        else:
            #Save white img
            background = 255 * np.ones(image.shape)
            save_img(background, out_folder, img)


def save_img(img2save, output_dir, picture_name):
    cv2.imwrite(os.path.join(output_dir, picture_name.split(".")[0]+".png"), img2save)  #

if __name__ == '__main__':
    #############PARAMETERS###############3
    ##AFFECTNET####
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-r', '--data_root', type=str, required=True,
                        help='Root path with the original images')
    parser.add_argument('-l', '--landmark_root_npy', type=str,
                        help='Root path with the landmarks npy files')
    parser.add_argument('-frm', '--part2rm', type=str,
                        help="Part of the face to remove. [OPTIONS: 'mouth', 'inner_mouth', 'right_eyebrow',"
                                                        "'left_eyebrow', 'right_eye', 'left_eye', 'nose', 'jaw'")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='Root path to savenew images')

    args = parser.parse_args()

    # for folder in folders:
    #     create_removed_part_img(folder, original_imgs, input_dir_npy_landm, output_dir, part2remove)
    ### CODE
    folders = os.listdir(args.data_root)
    obtain_landmarks_parallel(folders, args.data_root, args.landmark_root_npy, args.out_dir, args.part2rm)

    #Example parameters: -o
    # /.../AFFECTNET/Landmarks-No-Jaw
    # -l
    # .../AFFECTNET/LANDMARKS_dlib_MTCNN_npy_PRUEBA
    # -r
    # .../AFFECTNET/Manually_Annotated_compressed/Manually_Annotated_Images
    # -frm
    # '["nose","jaw"]'



