import os
from PIL import Image
import time, multiprocessing
from functools import partial
import numpy as np
from src.utils import image_utils
import argparse

def convert2grayAndResize(img_subfolder, in_path_landmarks,in_path_img_original,out_path_imgs, newSize, L2):
    in_path_subf = os.path.join(in_path_landmarks, img_subfolder)
    for img_name in os.listdir(in_path_subf):
        if (os.path.exists(os.path.join(out_path_imgs, img_subfolder, img_name))): return
        if L2 == "dilation":
            img_landm = np.array(Image.open(os.path.join(in_path_landmarks, img_subfolder, img_name)).convert("L"))
            imgPil = Image.fromarray(image_utils.create_dilation(img_landm))
        elif L2 == "soften":
            img_landm = np.array(Image.open(os.path.join(in_path_landmarks, img_subfolder, img_name)).convert("L"))
            img2 = image_utils.soften_img(img_landm, threshold=int(0.15*255))
            imgPil = Image.fromarray((img2).astype(np.uint8))
        elif L2== "collage":
            img_landm = np.array(Image.open(os.path.join(in_path_landmarks, img_subfolder, img_name)))/255
            img_original = np.array(Image.open(os.path.join(in_path_img_original, img_subfolder, img_name)))/255
            img3 = (img_landm * img_original)
            imgPil = Image.fromarray((img3*255).astype(np.uint8))

        grayScale = image_utils.convert2grayscale(imgPil)
        imgResized = image_utils.resize(grayScale, new_size=newSize)
        os.makedirs(os.path.join(out_path_imgs, img_subfolder), exist_ok=True)
        imgResized.save(os.path.join(out_path_imgs, img_subfolder, img_name.rsplit(".")[0] + ".png"))




def process_imgs_parallel(list_folders, in_path_imgs, in_path_imgs2, out_path_imgs, newSize, L2):
    """
    Extract frames in a parallel way using ffmpeg or opencv funcions
    """
    start_time = time.time()
    pool = multiprocessing.Pool()  # processes = 7

    landmarks_extractor = partial(convert2grayAndResize,  # get_npy_with_previous_clip_AVEC2019
                            in_path_imgs=in_path_imgs,
                            in_path_imgs2=in_path_imgs2,
                            out_path_imgs=out_path_imgs,
                            newSize = newSize,
                            L2=L2)

    pool.map(landmarks_extractor, list_folders)

    pool.close()
    pool.join()
    final_time = (time.time() - start_time)
    print("--- %s Data preparation TIME IN min ---" % (final_time / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-r', '--in_path_imgs_original', type=str, required=False,
                        help='Root path with the original images')
    parser.add_argument('-l', '--landmark_root', type=str, required=True,
                        help='Root path with the landmarks')
    parser.add_argument('-imgSize', '--img_size', type=int,
                        help='Output size of images',
                        default=48)
    parser.add_argument('-m', '--modality', type=str,
                        help='Choose the architecture of the model (collage, dilation, soften)',
                        default="soften")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='Root path to save generated images')

    args = parser.parse_args()


    # in_path_imgs_landm = ../LANDMARKS_dlib_MTCNN_PRUEBA" # Images to soften or dilate
    # in_path_imgs_original = "../Manually_Annotated_Images" # Landmarks to make collage
    # out_path_imgs = '../AFFECTNET/COLLAGE' # Output dir

    os.makedirs(args.out_dir, exist_ok=True)
    # for img_subfolder in os.listdir(in_path_imgs_landm):
    #     convert2grayAndResize(img_subfolder, in_path_imgs_landm,in_path_imgs_original, out_path_imgs, newSize, L2)
    #Parallel:
    list_folders = os.listdir(args.landmark_root)
    process_imgs_parallel(list_folders, args.landmark_root,args.in_path_imgs_original, args.out_dir,
                          (args.imgSize, args.imgSize), args.modality)
