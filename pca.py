    # O clasă pentru implementarea PCA (Analiza Componentelor Principale)

import numpy as np

class PCA:
    def __init__(self, X):  # presupunem că X este o matrice numpy standardizată
        self.X = X
        # Calculăm matricea de covarianță-variabilitate pentru X
        self.Cov = np.cov(m=X, rowvar=False)  # variabilele sunt pe coloane
        print(self.Cov.shape)
        # Extragem valorile proprii și vectorii proprii pentru matricea de covarianță
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(a=self.Cov)
        print(self.eigenvalues, self.eigenvalues.shape)
        print(self.eigenvectors.shape)
        # Sortăm valorile proprii în ordine descrescătoare, împreună cu vectorii proprii
        k_desc = [k for k in reversed(np.argsort(self.eigenvalues))]
        print(k_desc, type(k_desc))
        self.alpha = self.eigenvalues[k_desc]
        self.A = self.eigenvectors[:, k_desc]

        # Regularizarea vectorilor proprii
        for j in range(self.A.shape[1]):
            minCol = np.min(a=self.A[:, j], axis=0)  # variabilele sunt pe coloane
            maxCol = np.max(a=self.A[:, j], axis=0)
            if np.abs(minCol) > np.abs(maxCol):
                # Înmulțirea unui vector propriu cu o constantă nu schimbă natura acestuia
                self.A[:, j] = (-1) * self.A[:, j]

        # Calculăm componentele principale
        self.C = self.X @ self.A

        # Calculăm corelația între variabilele observate și componentele principale
        # cunoscută și sub numele de încărcături factoriale
        self.Rxc = self.A * np.sqrt(self.alpha)

        # Calculăm pătratul componentelor principale
        self.C2 = self.C * self.C

    def getEigenValues(self):
        # Returnăm valorile proprii
        return self.alpha

    def getEigenVectors(self):
        # Returnăm vectorii proprii
        return self.A

    def getPrinComp(self):
        # Returnăm componentele principale
        return self.C

    def getFactorLoadings(self):
        # Returnăm încărcăturile factoriale
        return self.Rxc

    def getScores(self):
        # Returnăm scorurile (componentele principale normalizate)
        return self.C / np.sqrt(self.alpha)

    def getQualObs(self):
        # Calculăm calitatea reprezentării observațiilor
        SL = np.sum(a=self.C2, axis=1)  # suma pe linii
        return np.transpose(self.C2.T / SL)

    def getContribObs(self):
        # Calculăm contribuția observațiilor la varianța axelor
        return self.C2 / (self.X.shape[0] * self.alpha)

    def getCommon(self):
        # Calculăm comunalitățile (varianța comună explicată de componente)
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1)  # suma cumulativă pe linii
