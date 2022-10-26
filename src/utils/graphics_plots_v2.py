import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def compare_models2colors(accuracies_red, accuracies_green, name_models, N_samples, title="", figsize=(8,5)):
    x = range(len(accuracies_red))
    y_red = accuracies_red
    dy_red = []
    x_green = range(len(accuracies_red), len(accuracies_red)+len(accuracies_green))
    y_green = accuracies_green
    dy_green = []
    for i_model in range(len(accuracies_red)):
        CI = calculate_CI(accuracies_red[i_model], N_samples)
        dy_red.append(CI)
        print("CI: ", str(CI))
    for i_model in range(len(accuracies_green)):
        CI = calculate_CI(accuracies_green[i_model], N_samples)
        dy_green.append(CI)
        print("CI: ", str(CI))

    fig, ax = plt.subplots(figsize=figsize)  # figsize=(13,10)
    plt.bar(x, y_red, yerr = dy_red, align='center',alpha=0.5, color = 'red', ecolor='red', capsize=10)
    plt.errorbar(x, y_red, yerr=dy_red, fmt='o', color='black',
                 ecolor='red', elinewidth=3, capsize=0)
    plt.bar(x_green, y_green, yerr=dy_green, align='center', alpha=0.5, color='green', ecolor='green', capsize=10)

    plt.errorbar(x_green, y_green, yerr=dy_green, fmt='o', color='black',
                 ecolor='green', elinewidth=3, capsize=0)
    # plt.xlabel("Models")
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
    ax.set_xticks(range(len(accuracies_red+accuracies_green)))
    # Set ticks labels for x-axis
    ax.set_xticklabels(name_models, rotation='0')
    ax.set(ylim=[72,77])

    ax.grid(b=True, which='major', axis="both", color='k', linestyle='--', linewidth=0.5)
    plt.title(title)
    plt.show()


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
    #FER-2013:
    N_samples = 31885
    accuracies_noTL = [72.92,74.20,73.64,74.93]
    accuracies_TL = [74.52,74.69,74.90,76.01]
    name_models = ["Baseline-STN", "Landm-STN\nBinary masks","Landm-STN\nSoft masks", "Saliency-STN",
                   "TL-Baseline-STN", "TL-Landm-STN\nBinary masks","TL-Landm-STN\nSoft masks", "TL-Saliency-STN"]

    title = "FER2013 - 5CV of 31.885 imgs"
    figsize=(12,6)
    compare_models2colors(accuracies_noTL, accuracies_TL, name_models, N_samples, title=title, figsize=figsize)

    #AffectNet
    N_samples = 325239
    accuracies = [69.78,70.37,70.57,70.72,70.53, 70.60, 66.47, 68.60]
    name_models = ["Simple-CNN","Baseline-STN", "Landm-STN\nBinary masks v1","Landm-STN\nSoft masks",
                   "Dilation-STN","Saliency-STN","Patches-STN","Weighted-STN"]
    figsize = (10, 6)
    title = "AffectNet - 5CV of 325.239 imgs"

    #compare_models(accuracies, name_models, N_samples, title=title, figsize=figsize)


