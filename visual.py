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

def init_dataset(opt):
    if opt.dataset == 0:
        # dataset = pd.read_csv("data2visual/omniglot_modelout_highloss.csv")
        dataset = pd.read_csv("data2visual/omniglot_modelout_lowloss.csv")
    elif opt.dataset == 1:
        dataset = pd.read_csv("data2visual/omniglot_test.csv")
        # havent finish yet
    elif opt.dataset == 2:
        dataset = pd.read_csv("data2visual/mnist_test.csv")
    else:
        raise(Exception("No such dataset!!"))

    dataset['y'] = dataset['label']
    dataset['label'] = dataset['label'].apply(lambda i: str(i))
    return dataset



def t_sne(df):
    feat_cols = dataset.columns.difference(['label', 'y'])
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
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


def load_omniglot_data(opt):
    # ordered_loss_dict = torch.load('ordered_loss_dict.pt', map_location=lambda storage, loc: storage)
    # print(ordered_loss_dict)

    # top = ordered_loss_dict.popitem()
    # top = np.array(top[0])
    # low = np.array(next(iter(ordered_loss_dict)))

    top = torch.load('x_y_high.pt', map_location=lambda storage, loc: storage)
    low = torch.load('x_y_low.pt', map_location=lambda storage, loc: storage)

    # load class directory
    # class_dict = np.load("idx.npy")
    # class_dict = class_dict.item()

    # img_data = []
    # img_label = []

    # for key, value in class_dict.items():
    #     if value in top:
    #         img_label.append(value)
    #         img_list = load_img(key)
    #         img_data.append(img_list)

    # df_data = []
    # for idx in range(len(img_label)):
    #     for img_idx in range(len(img_data[idx])):
    #         img_data[idx][img_idx].insert(0, img_label[idx])
    #         df_data.append(img_data[idx][img_idx])


    # model_out = np.array(top[0].detach().numpy()).tolist()
    # y = np.array(top[1].detach().numpy()).tolist()

    model_out = np.array(low[0].detach().numpy()).tolist()
    y = np.array(low[1].detach().numpy()).tolist()

    df_data = []
    for out, label in zip(model_out, y):
        out.insert(0, label)
        df_data.append(out)

    # top_data = pd.DataFrame(df_data)
    # top_data = top_data.rename(columns = {0:'label'})

    # top_data.to_csv("data2visual/omniglot_modelout_highloss.csv")

    low_data = pd.DataFrame(df_data)
    low_data = low_data.rename(columns = {0:'label'})

    low_data.to_csv("data2visual/omniglot_modelout_lowloss.csv")

    return low_data


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
    options = get_parser().parse_args()
    # load_omniglot_data(options)

    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    dataset = init_dataset(options)
    # rndperm = np.random.permutation(dataset.shape[0])
    # dataset = dataset.loc[rndperm[:1000],:]

    # t_sne(dataset)
    pca(dataset)

    # find_class(175)
