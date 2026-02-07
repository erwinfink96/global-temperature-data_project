import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


# Relativer Pfad zur CSV-Datei
project_dir = os.path.dirname(os.path.abspath(__file__))  # Ordner, in dem das Skript liegt
file_path = os.path.join(project_dir, 'GLB.Ts+dSSTnew.csv')



#Lade die CSV-Datei
df = pd.read_csv(file_path, skiprows=1)
#skiprows=1: die Überschrift des Datensatzes wird nicht miteingelesen


# nur erste Zeile entfernen, Index zurücksetzen
df = df.iloc[1:].reset_index(drop=True)

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

corr = df[months].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap="coolwarm",
    vmin=0.84,
    vmax=1.00,
    annot=True,        # ← Zahlen anzeigen
    fmt=".2f",         # ← 2 Nachkommastellen
    square=True,
    


)

plt.title("Korrelations-Heatmap der Monate")
plt.show()


# Jänner-Werte extrahieren
jan_values = df['Jan'].dropna().values

plt.figure(figsize=(4,6))
plt.boxplot(jan_values, vert=True)
plt.ylabel("Temperatur-Anomalie (°C)")
plt.title("Jänner: globale Land-Ozean-Temperaturanomalien")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()



# Jänner-Werte
jan_values = df['Jan'].dropna().values

plt.figure(figsize=(6,4))
plt.hist(jan_values, bins=20)
plt.xlabel("Temperatur-Anomalie (°C)")
plt.ylabel("Häufigkeit")
plt.title("Histogramm: Jänner Temperatur-Anomalien")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()





# -------------------------
# Jänner-Daten vorbereiten
# -------------------------
jan = df[['Year', 'Jul']].dropna()
jan['Year'] = jan['Year'].astype(int)

# Frühe und späte Periode definieren
jan_early = jan[(jan['Year'] >= 1880) & (jan['Year'] <= 1950)]['Jul']
jan_late  = jan[(jan['Year'] >= 1980) & (jan['Year'] <= 2025)]['Jul']

# -------------------------
# Statistik vergleichen
# -------------------------
print("Frühe Periode (1880–1950):")
print(" Mittelwert:", jan_early.mean())
print(" Std-Abweichung:", jan_early.std())

print("\nSpäte Periode (1980–2025):")
print(" Mittelwert:", jan_late.mean())
print(" Std-Abweichung:", jan_late.std())

# -------------------------
# Boxplot zur Visualisierung
# -------------------------
plt.figure(figsize=(6,5))
plt.boxplot([jan_early, jan_late], labels=['1880–1950', '1980–2025'])
plt.ylabel("Temperatur-Anomalie (°C)")
plt.title("Vergleich der Julianomalien 1880-1950 und 1980-2025")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()