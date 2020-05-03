import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

from parser_util import get_parser

import torch
from collections import OrderedDict

from PIL import Image

'''
https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
'''

def init_dataset(mode="top"):
    if mode == "top":
        dataset = pd.read_csv("data2visual/omniglot_test_high_loss.csv")
    else:
        dataset = pd.read_csv("data2visual/omniglot_test_low_loss.csv")

    dataset['y'] = dataset['label']
    dataset['label'] = dataset['label'].apply(lambda i: str(i))
    return dataset


def t_sne(df):
    feat_cols = dataset.columns.difference(['label', 'y'])
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure()
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 5),#10 if mnist
        data=df,
        legend="full",
        alpha=0.3
    )

    plt.show()


def pca(df):
    feat_cols = dataset.columns.difference(['label', 'y'])
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure()
    sns.scatterplot(
        x="pca-one",
        y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 5),
        data=df,
        legend="full",
        alpha=0.3
    )

    # For a 3d-version of the same plot
    # ax = plt.figure().gca(projection='3d')
    # ax.scatter(
    #     xs=df.loc[rndperm,:]["pca-one"],
    #     ys=df.loc[rndperm,:]["pca-two"],
    #     zs=df.loc[rndperm,:]["pca-three"],
    #     c=df.loc[rndperm,:]["y"],
    #     cmap='tab10'
    # )
    # ax.set_xlabel('pca-one')
    # ax.set_ylabel('pca-two')
    # ax.set_zlabel('pca-three')
    plt.show()


def load_omniglot_data(mode="top"):
    ordered_loss_dict = torch.load('ordered_loss_dict.pt', map_location=lambda storage, loc: storage)

    top = ordered_loss_dict.popitem()
    top = np.array(top[0])
    low = np.array(next(iter(ordered_loss_dict)))


    # load class directory
    class_dict = np.load("idx.npy")
    class_dict = class_dict.item()

    img_data = []
    img_label = []

    if mode == "top":
        for key, value in class_dict.items():
            if value in top:
                img_label.append(value)
                img_list = load_img(key)
                img_data.append(img_list)
    else:
        for key, value in class_dict.items():
            if value in low:
                img_label.append(value)
                img_list = load_img(key)
                img_data.append(img_list)

    df_data = []
    for idx in range(len(img_label)):
        for img_idx in range(len(img_data[idx])):
            img_data[idx][img_idx].insert(0, img_label[idx])
            df_data.append(img_data[idx][img_idx])


    if mode == "top":
        top_data = pd.DataFrame(df_data)
        top_data = top_data.rename(columns = {0:'label'})
        top_data.to_csv("data2visual/omniglot_test_high_loss.csv")
    else:
        low_data = pd.DataFrame(df_data)
        low_data = low_data.rename(columns = {0:'label'})
        low_data.to_csv("data2visual/omniglot_test_low_loss.csv")



def load_img(path):
    path = "../dataset/data/" + path
    path, rot = path.split(os.sep + 'rot')

    img_paths = os.listdir(path)
    img_list = []
    for img_path in img_paths:
        x = Image.open(path+os.sep+img_path)
        x = x.rotate(float(rot))
        x = x.resize((28, 28))

        shape = -1
        x = np.array(x, np.float32, copy=False)
        x = 1.0 - torch.from_numpy(x)
        x = x.transpose(0, 1).contiguous().view(shape)
        x = np.array(x).ravel().tolist()

        img_list.append(x)

    return img_list


def find_class(num):
    class_dict = np.load("idx.npy")
    class_dict = class_dict.item()

    for key, value in class_dict.items():
        if value == num:
            print(key)
            break
    # e.g. 175 -> Oriya/character30/rot270


if __name__ == "__main__":
    
    mode = "top" #or low

    load_omniglot_data(mode)
    dataset = init_dataset(mode)


    t_sne(dataset)
    # pca(dataset)

    # find_class(175)
