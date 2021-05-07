import os
from PIL import Image
import argparse

def refactor_images_AffectNet(list_imgs, input_path, extensions2change=["bmp", "BMP","tif", "TIF", "tiff", "TIFF"]):
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
    args = parser.parse_args()
    folders_AffectNet = os.listdir(args.data_root)
    for folder in folders_AffectNet:
        in_folder_path = os.path.join(args.data_root, folder)
        refactor_images_AffectNet(os.listdir(in_folder_path), in_folder_path)



