import pandas as pd
import numpy as np
import graficeHCA as graphics
import utils as utils
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl

try:
    # Numele fișierului de intrare
    fileName = 'dataIN/InfantMortality.csv'

    # Setăm un avertisment pentru mai mult de 50 de figuri deschise
    mpl.rcParams['figure.max_open_warning'] = 50

    # Lista opțiunilor pentru rularea codului
    # Păstrăm în listă doar opțiunile dorite
    drawing_options = ['Graficul partiției în axele principale',
                       # 'Plotarea histogramelor',
                       'Gruparea variabilelor']
    discriminant_axes = (drawing_options.
                         __contains__('Graficul partiției în axele principale'))
    histograms = (drawing_options.
                  __contains__('Plotarea histogramelor'))
    variable_grouping = (drawing_options.
                         __contains__('Gruparea variabilelor'))

    # Citim tabelul din fișierul CSV
    table = pd.read_csv(fileName, index_col=0)

    # Extragem variabilele
    vars = table.columns.values[1:]
    print(vars)

    # Extragem observațiile
    obs = table.index.values
    print(obs, type(obs))

    # Creăm matricea de date X
    X = table[vars].values
    print(X, type(X))

    # Standardizăm matricea X
    Xstd = utils.standardise(X)
    print(Xstd)

    # Crearea ierarhiei instanțelor
    methods = list(hclust._LINKAGE_METHODS)
    metrics = hdist._METRICS_NAMES
    print('Metode: ', methods)
    print('Metrici: ', metrics)

    # Alegerea metodei și a metricii
    method = methods[5]  # ward
    distance = metrics[3]  # citiblock

    # Verificăm dacă metoda necesită metrica euclidiană
    if method in ['ward', 'centroid', 'median', 'weighted']:
        distance = 'euclidean'

    # Creăm matricea de legături pentru clustere
    h = hclust.linkage(Xstd, method=method, metric=distance)

    # Identificarea partiției de stabilitate maximă
    m = np.shape(h)[0]  # Numărul maxim de joncțiuni
    k = m - np.argmax(h[1:m, 2] - h[:(m - 1), 2])  # Numărul optim de clustere

    # Identificarea clusterelor din partiția optimă
    g_max, codes = utils.cluster_distribution(h, k)

    # Afișăm clusterul
    utils.cluster_display(g_max, table.index, 'Țară',
                          './dataOUT/Partitia_max.csv')

    # Salvăm distribuția clusterelor într-un fișier CSV
    utils.cluster_save(g=g_max, row_labels=table.index.values,
                       col_label='Cluster',
                       file_name='PartitiaOptima.csv')

    t_1, j_1, m_1 = utils.threshold(h)
    print('Prag=', t_1, 'Joncțiunea cu diferența maximă=', j_1,
          'Număr de joncțiuni=', m_1)

    # Determinăm culorile pentru clustere
    color_clusters = utils.color_clusters(h, k, codes)

    # Generăm dendograma observațiilor
    graphics.dendrogram(h, labels=obs,
                        title='Clusterizarea observațiilor. Partiția cu stabilitate maximă. '
                              'Metoda: ' + method + ' Metrica: ' + distance,
                        colors=color_clusters, threshold=t_1)

    # Plotăm clusterele în axele principale
    if discriminant_axes:
        model_pca = dec.PCA(n_components=2)
        z = model_pca.fit_transform(Xstd)
        groups = list(set(g_max))
        graphics.plot_clusters(z[:, 0], z[:, 1], g_max, groups,
                               labels=obs, title='Partiția optimă')

    # Generăm histogramele pentru variabile
    if histograms:
        for v in vars:
            graphics.histograms(table[v].values, g_max, var=v)

    # Gruparea variabilelor
    if variable_grouping:
        method_v = 'average'
        distance_v = 'correlation'
        h2 = hclust.linkage(X.transpose(), method=method_v, metric=distance_v)
        t_2, j_2, m_2 = utils.threshold(h2)
        print('Prag=', t_2, 'Joncțiunea cu diferența maximă=', j_2,
              'Număr de joncțiuni=', m_2)
        graphics.dendrogram(h2, labels=vars,
                            title="Clusterizarea variabilelor. Metoda: " + method_v +
                                  " Metrica: " + distance_v, threshold=t_2)

    n = np.shape(X)[0]

    # Creăm o listă de selecții pentru partiții
    list_selections = [str(i) + ' clustere' for i in range(2, n - 1)]
    partitions = list_selections[0:5]

    # Creăm DataFrame cu partiția de stabilitate maximă și partițiile selectate
    t_partitions = pd.DataFrame(index=table.index)
    t_partitions['Partitia_max'] = g_max
    for v in partitions:
        k = list_selections.index(v) + 2  # Numărul dorit de clustere
        g, codes = utils.cluster_distribution(h, k)

        # Salvăm partiția
        utils.cluster_save(g=g, row_labels=table.index.values,
                           col_label='Cluster', file_name='Partitia_' + str(k) + '.csv')

        color_clusters = utils.color_clusters(h, k, codes)
        graphics.dendrogram(h, obs,
                            title='Partiția cu ' + v,
                            colors=color_clusters, threshold=t_1)
        t_partitions['P_' + v] = g

        # Plotăm clusterele în axele principale pentru fiecare partiție
        if discriminant_axes:
            z = model_pca.fit_transform(Xstd)
            groups = list(set(g))
            graphics.plot_clusters(z[:, 0], z[:, 1], g, groups, labels=obs,
                                   title='Partiția cu ' + v)

        # Generăm histogramele pentru fiecare variabilă
        if histograms:
            for v in vars:
                graphics.histograms(table[v].values, g, var=v)

    # Salvăm toate partițiile într-un fișier CSV
    t_partitions.to_csv('./dataOUT/PartitiaTotal.csv')

    # Afișăm toate graficele generate
    graphics.show()

except Exception as ex:
    # Gestionăm și afișăm eventualele erori
    print("Eroare!", ex)
