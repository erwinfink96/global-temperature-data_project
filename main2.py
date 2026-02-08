from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import os 
import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


# Relativer Pfad zur CSV-Datei
project_dir = os.path.dirname(os.path.abspath(__file__))  # Ordner, in dem das Skript liegt
file_path = os.path.join(project_dir, 'GLB.Ts+dSSTnew.csv')



#Lade die CSV-Datei
df = pd.read_csv(file_path, skiprows=1)
#skiprows=1: die Überschrift des Datensatzes wird nicht miteingelesen


# nur erste Zeile entfernen, Index zurücksetzen
df = df.iloc[1:].reset_index(drop=True)
print(df.loc[df["Year"] == 2025, "J-D"])


# Monate definiereen 
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
x = df[months].values
y = df['J-D'].values

# Feature-Importance berechen (Wie wichtig sind die einzelnen Monate?)
rf = RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(x, y)

importances = rf.feature_importances_        # array mit 12 Zahlen 
for m, imp in sorted(zip(months, importances), key=lambda z: z[1], reverse=True):
    print(f"{m}: {imp:.3f}")

# SHAP-Wert-Berechnung 
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(x)

shap.summary_plot(shap_values, x, feature_names=months)



#Residuuenanalyse 
y_pred = rf.predict(x)
residuals = y - y_pred                     #Abweichung vom tatsächlichen Jahresdurchschnitt

plt.scatter(df['Year'], residuals, color = "pink")
plt.axhline(0)
plt.xlabel("Year")
plt.ylabel("Residual")
plt.show()





# Clustererkennung
x = df[months].values   # Monatsprofile
years = df["Year"].values


# KMeans (Machine Learning)
kmeans = KMeans(n_clusters=4, random_state=0)
df["cluster"] = kmeans.fit_predict(x)


# Cluster-Plot
plt.figure(figsize=(9,5))

# Alle Monatswerte plotten
for row in x:
    plt.scatter(months, row, color="gray", alpha=0.1)

# Clusterzentren plotten 
for i, center in enumerate(kmeans.cluster_centers_):
    plt.plot(months, center, marker='o', label=f"Cluster {i}")

plt.xlabel("Monat")
plt.ylabel("Temperatur-Anomalie [°C]")
plt.title("Typische Jahresprofile (KMeans)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Jahre nach Cluster einfärben
plt.figure(figsize=(10,4))

for i in range(4):
    mask = df["cluster"] == i
    plt.scatter(df.loc[mask, "Year"], df.loc[mask, "J-D"],
                label=f"Cluster {i}", s=25)

plt.xlabel("Jahr")
plt.ylabel("Jahresanomalie [°C]")
plt.title("Jahre nach Monatsprofil-Clustern")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# x = df[months].values
# y = df['J-D'].values

# # Ist y der Mittelwert der Monate?
# y_from_months = x.mean(axis=1)
# print("max abs diff:", np.nanmax(np.abs(y - y_from_months)))

# # Std und Korrelation
# stds = np.nanstd(x, axis=0)
# corrs = [np.corrcoef(x[:,i], y)[0,1] for i in range(12)]
# df_stats = pd.DataFrame({
#     'month': months,
#     'std': stds,
#     'corr_with_y': corrs
# }).sort_values('std', ascending=False)

# print(df_stats.to_string(index=False))