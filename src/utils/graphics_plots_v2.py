import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def compare_models(accuracies, name_models, N_samples, title="", figsize=(8,5)):
    x = range(len(accuracies))
    y = accuracies
    dy = []
    for i_model in range(len(accuracies)):
        CI = calculate_CI(accuracies[i_model], N_samples)
        dy.append(CI)
        print("CI: ", str(CI))

    fig, ax = plt.subplots(figsize=figsize) #figsize=(13,10)
    plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
                 ecolor='red', elinewidth=3, capsize=0)
    #plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")

    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.minorticks_on()
    ax.grid(which='major', color='black', linestyle='--')
    ax.grid(which='minor', color='gray', linestyle=':')

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(name_models, rotation='0')

    ax.grid(b=True, which='major',axis="both", color='k', linestyle='--', linewidth=0.5)
    plt.title(title)
    plt.show()

def calculate_CI(accuracy, N_samples):
    return 1.96*np.sqrt((accuracy*(100-accuracy))/N_samples)


if __name__ == '__main__':
    #FER2013 - 5CV - polarity:
    # N_samples = 27557
    # accuracies = [70.72, 71.20, 71.90]
    # name_models = ["Original\n48x48", "Original+Landmarks\n48x48",  "Original+Saliency\n48x48"]
    # title = "FER2013 - 5CV of 27.557 imgs - polarity"
    # compare_models(accuracies, name_models, N_samples, title=title)
    # FER2013 - 5CV - polarity TL:
    # N_samples = 27557
    # accuracies = [70.02,70.52,69.25,71.89]
    # name_models = ["Baseline\n48x48","Original\n48x48", "Original+Landmarks\n48x48", "Original+Saliency\n48x48"]
    # title = "FER2013 - 5CV of 27.557 imgs - polarity - Transfer Learning"
    #NO TRANSFER LEARNING
    N_samples = 31885
    accuracies = [73.40,72.92,74.20,73.64,74.93]
    name_models = ["Simple-CNN","Baseline-ST", "Landm-ST\nBinary masks","Landm-ST\nSoft masks", "Saliency-ST"]

    title = "FER2013 - 5CV of 31.885 imgs - NO Transfer Learning"
    figsize=(10,6)
    # YES TRANSFER LEARNING
    # N_samples = 31885
    # accuracies = [70.18, 71.35, 72.56, 71.99, 72.37]  # 70.78,71.74,71.57,71.46
    # name_models = ["Simple-CNN", "Baseline-ST", "Landm-ST\nbinary masks", "Landm-ST\nSoft masks", "Saliency-ST"]
    #
    # title = "FER2013 - 5CV of 31.885 imgs - Transfer Learning"
    # figsize = (10, 6)
    #
    # # "TL-Baseline","TL-Original", "TL-Landm","TL-Landm\nSoft(L2)", "TL-Saliency"]
    #
    #BOTH:
    # N_samples = 31885
    # accuracies = [73.40,72.92,74.20,73.64,74.93,
    #               ]
    # name_models = ["Simple-CNN","Baseline-ST", "Landm-ST\nBinary masks","Landm-ST\nSoft masks", "Saliency-ST",
    #                "TL-Simple-CNN","TL-Baseline-ST", "TL-Landm-ST\nBinary masks","TL-Landm-ST\nSoft masks", "TL-Saliency-ST"]
    #
    # title = "FER2013 - 5CV of 31.885 imgs"
    # figsize=(10,6)



    # N_samples = 325239
    # accuracies = [69.54,70.07,70.57,70.03, 69.31, 67.67, 69.31, 70.62]
    # name_models = ["Baseline","ST-Original", "ST-Landmarks",
    #                "ST-Soft\nLandmarks L2","Collage\nLand","Collage\nSoft L2",
    #                "ST-Dilation","ST-Saliency"]
    # figsize = (10, 6)
    # title = "AffectNet - 5CV of 325.239 imgs"


    compare_models(accuracies, name_models, N_samples, title=title, figsize=figsize)
    # FER ALL:
    # accuracies = [70.72, 71.20, 71.90, 72.26, 72.26, 74.26]
    # name_models = ["Original\n48x48", "Original+Landmarks\n48x48",  "Original+Saliency\n48x48",
    #                "Original\n48x48-TL", "Original+Landmarks\n48x48-TL", "Original+Saliency\n48x48-TL"]
    # title = "FER2013 - 5CV of 27.557 imgs - polarity - ALL"
    # compare_models(accuracies, name_models, N_samples, title=title)
