import os
import numpy as np
import pandas as pd
import cv2

from torch import nn
import scipy.io as scio

if __name__ == '__main__':
    dataFile = r"Twitter_LDL/twitter_ldl_config.mat"
    data = scio.loadmat(dataFile)
    # print(data["imgname"])
    # print(data["train_ind"])
    # print(data["test_ind"])
    img_names = []
    for imgname, test in zip(data["imgname"], data["train_ind"]):
        if test[0]:
            img_names.append(imgname[0][0])
    df = pd.read_csv("Twitter_LDL/ground_truth.txt", sep=" ")
    # print(df.columns)
    # df_train = df[df["images-name"].isin(img_names)]
    df = df[df["images-name"].isin(img_names)].iloc[:, :-1]
    print(df.shape)
    df.to_csv("Twitter_LDL/train_ground_truth.txt", sep=" ", index=None)

    # with open("./Emotion6/ground_truth.txt", "r") as f1, open("./Emotion6/train_ground_truth.txt", "w") as f2:
    #     print(*f1.readlines()[:-400], sep="", file=f2)
