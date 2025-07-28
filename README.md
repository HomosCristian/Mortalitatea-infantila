# Proiect de Analiză a Datelor despre Mortalitatea Infantilă

## Prezentare Generală
Acest proiect are ca scop analizarea factorilor socio-economici legați de mortalitatea infantilă în țările membre ale Uniunii Europene (UE). Analiza folosește tehnici statistice avansate pentru a extrage tipare semnificative din date și pentru a le prezenta într-un mod interpretabil.

### Caracteristici principale:
1. **Analiza Componentelor Principale (PCA):** Reduce dimensiunea datelor păstrând variația semnificativă.
2. **Analiza Cluster Ierarhică (HCA):** Grupează țările în clustere pe baza similitudinilor socio-economice.
3. **Vizualizare a datelor:** Generează vizualizări interactive și statice pentru a ilustra concluziile.

---

## Metode utilizate

### **1. Analiza Componentelor Principale (PCA):**
- **Obiectiv:** Reducerea dimensiunii datasetului, păstrând cele mai importante informații.
- **Rezultate:**
  - Valori proprii și vectori proprii.
  - Componente principale.
  - Încărcături factoriale.
  - Vizualizări: Cerc de corelație și graficul variației explicate.

### **2. Analiza Cluster Ierarhică (HCA):**
- **Obiectiv:** Gruparea țărilor în clustere pe baza similitudinilor socio-economice.
- **Rezultate:**
  - Dendrogramă pentru vizualizarea ierarhiei clusterelor.
  - Atribuirea fiecărei țări la un cluster.

---

## Fișiere și directoare

### **Fișiere principale:**
- `main.py`: Punctul de intrare pentru rularea proiectului.
- `pca.py`: Implementarea Analizei Componentelor Principale.
- `hca.py`: Implementarea Analizei Cluster Ierarhice.
- `matriceStandardizata.py`: Preprocesarea datelor prin standardizare.

### **Module pentru vizualizări:**
- `grafice.py`: Generează grafice pentru rezultatele PCA.
- `graficeHCA.py`: Creează dendograme pentru HCA.
- `vizualizari.py`: Suportă vizualizări suplimentare.

### **Fișiere de utilități:**
- `utils.py`: Conține funcții ajutătoare pentru procesarea și validarea datelor.

### **Fișiere de date:**
- `InfantMortality.csv`: Setul de date de intrare care conține indicatori socio-economici pentru 27 de țări UE.
- `dataOUT/`: Director în care sunt salvate datele procesate și rezultatele analizei.

---


