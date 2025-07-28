import pandas as pd
import matriceStandardizata as sm
import pca as pca
import grafice as g

# Citim datele din fisierul CSV
table = pd.read_csv('./dataIN/InfantMortality.csv', index_col=0)
print(table)

# Crearea unei liste de variabile utile
vars = table.columns.values[0:]  # Selectam toate coloanele disponibile
print(vars, type(vars))

# Crearea unei liste de observații
obs = table.index.values  # Selectam indexul rândurilor (numele observațiilor)
print(obs, type(obs))

# Numărul de variabile
m = vars.shape[0]
print(m)

# Numărul de observații
n = len(obs)
print(n)

# Crearea matricei X cu variabilele observate
X = table[vars].values
print(X.shape, type(X))

# Standardizarea matricei X
Xstd = sm.standardize(X)
print(Xstd.shape)

# Salvăm matricea standardizată într-un fișier CSV
Xstd_df = pd.DataFrame(data=Xstd, index=obs, columns=vars)
print(Xstd_df)
Xstd_df.to_csv('./dataOUT/MatriceaStandardizata.csv')

# Instanțierea unui obiect PCA
modelPCA = pca.PCA(Xstd)
alpha = modelPCA.getEigenValues()

# Afișarea valorilor proprii (varianța explicată)
g.principalComponents(eigenvalues=alpha)

# Extragem componentele principale
prinComp = modelPCA.getPrinComp()

# Salvăm componentele principale într-un fișier CSV
components = ['C' + str(j + 1) for j in range(prinComp.shape[1])]
prinComp_df = pd.DataFrame(data=prinComp, index=obs, columns=components)
prinComp_df.to_csv('./dataOUT/PCA.csv')

# Extragem încărcăturile factorilor
factorLoadings = modelPCA.getFactorLoadings()
factorLoadings_df = pd.DataFrame(data=factorLoadings, index=vars, columns=components)
print(factorLoadings_df)

# Salvăm încărcăturile factorilor într-un fișier CSV
factorLoadings_df.to_csv('./dataOUT/IncarcaturaFactorilor.csv')

# Creăm un corelogram pentru încărcăturile factorilor
g.correlogram(matrix=factorLoadings_df, title='Corelograma încărcăturilor factorilor')

# Extragem scorurile
scores = modelPCA.getScores()
scores_df = pd.DataFrame(data=scores, index=obs, columns=components)

# Salvăm scorurile
scores_df.to_csv('./dataOUT/Scoruri.csv')
g.correlogram(matrix=scores_df, title='Corelograma scorurilor')

# Extragem calitatea reprezentării punctelor
qualObs = modelPCA.getQualObs()
qualObs_df = pd.DataFrame(data=qualObs, index=obs, columns=components)

# Salvăm calitatea reprezentării punctelor
qualObs_df.to_csv('./dataOUT/CalitateaPunctelor.csv')
g.correlogram(matrix=qualObs_df, title='Corelograma calității reprezentării punctelor')

# Extragem contribuția observațiilor la varianța axelor
contribObs = modelPCA.getContribObs()
contribObs_df = pd.DataFrame(data=contribObs, index=obs, columns=components)

# Salvăm contribuția observațiilor
contribObs_df.to_csv('./dataOUT/ContributiaObs.csv')
g.correlogram(matrix=contribObs_df, title="Corelograma contribuției observațiilor la varianța axelor")

# Extragem comunalitățile
common = modelPCA.getCommon()
common_df = pd.DataFrame(data=common, index=vars, columns=components)

# Salvăm comunalitățile
common_df.to_csv('./dataOUT/Comunalitati.csv')
g.correlogram(matrix=common_df, title='Corelograma comunalităților')

# Afișăm toate graficele
g.show()
