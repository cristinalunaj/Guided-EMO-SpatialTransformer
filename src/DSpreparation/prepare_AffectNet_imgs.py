import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
from PIL import Image
import argparse
import time, multiprocessing
from functools import partial
from src.utils import image_utils

def convert_imgs_parallel(list_folders, in_path_imgs, out_path_imgs, newSize=(48,48)):
    """
    Convert images to grayscale and resize (parallel)
    :param list_folders: List of the folders that contain the images to convert
    :param in_path_imgs: Root path where the folders of images are
    :param out_path_imgs: Output path to save generated images
    :param newSize: New size of the images (default 48x48)
    """
    start_time = time.time()
    pool = multiprocessing.Pool()  # processes = 7

    imgs_conversor = partial(image_utils.convert2grayAndResize,  # get_npy_with_previous_clip_AVEC2019
                            in_path_imgs=in_path_imgs
                            ,out_path_imgs=out_path_imgs,
                              newSize = newSize)

    pool.map(imgs_conversor, list_folders)

    pool.close()
    pool.join()
    final_time = (time.time() - start_time)
    print("--- %s Data preparation TIME IN min ---" % (final_time / 60))




def refactor_images_AffectNet(list_imgs, input_path, extensions2change=["bmp", "BMP","tif", "TIF", "tiff", "TIFF"]):
    """
    Change the extension of some images of AffectNet to .jpg (and save them in the same folder,
    removing the previous images with the 'wrong' extension)
        :param list_imgs: Name of images to check
        :param input_path: Root path where the images are
        :param extensions2change: Extensions to modify by .jpg
    """
    for img in list_imgs:
        path_img = os.path.join(input_path, img)
        extension = img.split(".")[-1]
        if(extension in extensions2change):
            print("MODIFY IMG: ", path_img)
            im = Image.open(path_img)
            rgb_im = im.convert('RGB')
            new_path_img = os.path.join(input_path, img.split(".")[0]+".jpg")
            rgb_im.save(new_path_img)
            #Remove previous image:
            os.remove(path_img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-r', '--data_root', type=str, required=True,
                        help='Root path with the images')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='Root path to save resized and grayscale images')
    parser.add_argument('-imgSize', '--img_size', type=int,
                        help='Output size of images',
                        default=48)
    parser.add_argument('-ds', '--dataset_name', type=str, help='Name of the dataset to process ["AffectNet","FER"]',
                        default='AffectNet')

    args = parser.parse_args()
    folders_orImages = os.listdir(args.data_root)
    #Change extensions of formats not recognized by saliency model: ("bmp", "BMP","tif", "TIF", "tiff", "TIFF")
    print("Checking images ...")
    if (args.dataset_name == "AffectNet"):
        for folder in folders_orImages:
            in_folder_path = os.path.join(args.data_root, folder)
            refactor_images_AffectNet(os.listdir(in_folder_path), in_folder_path)
        # Extract grayscale images and resize to 48x48
        print("Start image conversion to 48x48 & grayscale ...")
        os.makedirs(args.out_dir, exist_ok=True)
        convert_imgs_parallel(folders_orImages, args.data_root, args.out_dir, newSize=(args.img_size, args.img_size))
    else:
        refactor_images_AffectNet(folders_orImages, args.data_root)
        print("Start image conversion to 48x48 & grayscale ...")
        os.makedirs(args.out_dir, exist_ok=True)
        image_utils.convert2grayAndResize(img_subfolder=args.data_root.split("/")[-1],
                                          in_path_imgs = args.data_root.rsplit("/",1)[0], out_path_imgs = args.out_dir,newSize = (args.img_size, args.img_size))








