import numpy as np
import pandas as pd
import pandas.api.types as pdt
import graficeHCA as graphics


    # Calculăm media și deviația standard pe coloane
def standardise(x):
    # x - tabel de date, se așteaptă numpy.ndarray

    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    # Standardizăm matricea
    Xstd = (x - means) / stds
    return Xstd

    # Centrarea datelor prin scăderea mediei
def center(x):
    # x - tabel de date, se așteaptă numpy.ndarray

    means = np.mean(x, axis=0)
    return (x - means)

    # Regularizarea vectorilor proprii
def regularise(t, y=None):
    # t - tabel de vectori proprii,
    # se așteaptă numpy.ndarray sau pandas.DataFrame

    if isinstance(t, pd.DataFrame):
        for c in t.columns:
            minim = t[c].min()
            maxim = t[c].max()
            if abs(minim) > abs(maxim):
                t[c] = -t[c]
                if y is not None:
                    k = t.columns.get_loc(c)
                    y[:, k] = -y[:, k]
    if isinstance(t, np.ndarray):
        for i in range(np.shape(t)[1]):
            minim = np.min(t[:, i])
            maxim = np.max(t[:, i])
            if np.abs(minim) > np.abs(maxim):
                t[:, i] = -t[:, i]
    return None

    # Înlocuirea valorilor lipsă cu medie/mod
def replace_na_df(t):
    # t - pandas.DataFrame

    for c in t.columns:
        if pdt.is_numeric_dtype(t[c]):
            if t[c].isna().any():
                avg = t[c].mean()
                t[c] = t[c].fillna(avg)
        else:
            if t[c].isna().any():
                mode = t[c].mode()
                t[c] = t[c].fillna(mode[0])
    return None

    # Înlocuirea valorilor lipsă cu media
def replace_na(X):
    # X - numpy.ndarray

    means = np.nanmean(X, axis=0)
    k_nan = np.where(np.isnan(X))
    X[k_nan] = means[k_nan[1]]
    return None

    # Determinarea distribuției clusterelor
def cluster_distribution(h, k):
    # h - ierarhia, numpy.ndarray
    # k - numărul de clustere

    n = np.shape(h)[0] + 1
    g = np.arange(0, n)
    print('g: ', g)
    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g == k1] = n + i
        g[g == k2] = n + i
    g_ = pd.Categorical(g)
    return ['C' + str(i) for i in g_.codes], g_.codes

    # Calculul pragului pentru determinarea partiției cu stabilitate maximă
def threshold(h):
    m = np.shape(h)[0]
    print('m=', m)
    dist_1 = h[1:m, 2]
    dist_2 = h[0:m - 1, 2]
    diff = dist_1 - dist_2
    print('Diferențe:', diff)
    j = np.argmax(diff)
    print('j=', j)
    threshold = (h[j, 2] + h[j + 1, 2]) / 2
    return threshold, j, m

    # Afișarea distribuției clusterelor
def cluster_display(g, labels, label_names, file_name):
    g_ = np.array(g)
    groups = list(set(g))
    m = len(groups)
    table = pd.DataFrame(index=groups)
    clusters = np.full(shape=(m,), fill_value="",
                       dtype=np.chararray)
    for i in range(m):
        cluster = labels[g_ == groups[i]]
        cluster_str = ""
        for v in cluster:
            cluster_str += (v + " ")
        clusters[i] = cluster_str
    table[label_names] = clusters
    table.to_csv(file_name)

    # Salvarea distribuției clusterelor într-un fișier CSV
def cluster_save(g, row_labels, col_label, file_name):
    pairs = zip(row_labels, g)
    pairs_list = [g for g in pairs]
    g_dict = {k: v for (k, v) in pairs_list}
    g_df = pd.DataFrame.from_dict(data=g_dict,
                                  orient='index', columns=[col_label])
    print(g_df)
    g_df.to_csv('./dataOUT/' + file_name)

def color_clusters(h, k, codes):
    # h - ierarhia, numpy.ndarray
    # k - numărul de culori
    # codes - codurile clusterelor

    colors = np.array(graphics._COLORS)
    nr_colors = len(colors)
    m = np.shape(h)[0]
    n = m + 1
    cluster_colors = np.full(shape=(2 * n * 1,),
                             fill_value="", dtype=np.chararray)
    for i in range(n):
        cluster_colors[i] = colors[codes[i] % nr_colors]
    for i in range(m):
        k1 = int(h[i, 0])
        k2 = int(h[i, 1])
        if cluster_colors[k1] == cluster_colors[k2]:
            cluster_colors[n + i] = cluster_colors[k1]
        else:
            cluster_colors[n + i] = 'k'
    return cluster_colors
