import os
import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from PIL import Image
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def channels(img):
    compose = transforms.Compose([transforms.ToTensor()])
    return compose(img).shape[0]


def deal_img():
    folders = ["Emotion6", "Flickr_LDL", "Twitter_LDL"]
    t = ["train", "test", ]
    compose = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for folder in folders:
        for n in t:
            gt_path = os.path.join(folder, n + "_ground_truth.txt")
            df: pd.DataFrame = pd.read_csv(gt_path, sep=" ")
            images_name = df["images-name"]
            if folder == "Emotion6":
                l_list = [
                    "[prob. anger]",
                    "[prob. disgust]",
                    "[prob. fear]",
                    "[prob. joy]",
                    "[prob. sadness]",
                    "[prob. surprise]",
                    "[prob. neutral]",
                ]
            else:
                l_list = [
                    "Amusement",
                    "Awe",
                    "Contentment",
                    "Excitement",
                    "Anger",
                    "Disgust",
                    "Fear",
                    "Sadness",
                ]
            label_list = torch.tensor(df[l_list].values.astype(np.float32))
            a = torch.zeros([1, 3, 224, 224])
            for i, image in tqdm(enumerate(images_name), total=len(images_name)):
                image_path = os.path.join(folder, "images", image)
                img = Image.open(image_path)
                # if channels(img) == 1:
                #     print(image_path, n)
                b = torch.unsqueeze(compose(img), 0)
                a = torch.cat((a, b), 0)
            a = a[1:]
            data = {
                "data": a,
                "target": label_list,
            }
            save_path = os.path.join(folder, n + ".tf")
            torch.save(data, save_path)
            print(save_path, "已保存")


# Image.open("Emotion6/images")
if __name__ == '__main__':
    deal_img()
