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
from omniglot_dataset import OmniglotDataset
from miniImageNet_dataset import MiniImageNet

'''
https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
'''

def init_dataset(opt):
    if opt.dataset == 0:
        dataset = OmniglotDataset(mode='train', root=opt.dataset_root)
    elif opt.dataset == 1:
        dataset = MiniImageNet('train')
        # filename_label = pd.read_csv("miniImageNet/train.csv")
    elif opt.dataset == 2:
        dataset = pd.read_csv("data2visual/mnist_test.csv")
        dataset['y'] = dataset['label']
        dataset['label'] = dataset['label'].apply(lambda i: str(i))
    else:
        raise(Exception("No such dataset!!"))
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
        palette=sns.color_palette("hls", 10),
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
        palette=sns.color_palette("hls", 10),
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


if __name__ == "__main__":
    options = get_parser().parse_args()

    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    dataset = init_dataset(options)
    rndperm = np.random.permutation(dataset.shape[0])
    dataset = dataset.loc[rndperm[:1000],:]

    t_sne(dataset)
    # pca(dataset)
