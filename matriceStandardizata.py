import numpy as np

def standardize(X):  # presupunem că primim un obiect numpy.ndarray
    # Calculăm mediile pe coloane
    means = np.mean(a=X, axis=0)
    print(means.shape)

    # Calculăm deviațiile standard pe coloane (variabilele sunt pe coloane)
    stds = np.std(a=X, axis=0)

    # Returnăm matricea standardizată
    return (X - means) / stds
