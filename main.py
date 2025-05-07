#Importiere die nötigen Packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Relativer Pfad zur CSV-Datei
project_dir = os.path.dirname(os.path.abspath(__file__))  # Ordner, in dem das Skript liegt
file_path = os.path.join(project_dir, 'GLB.Ts+dSST.csv')



#Lade die CSV-Datei

#df= pd.read_csv(r'C:\Users\Erwin\Documents\Python_Projekt_2\GLB.Ts+dSST.csv',skiprows=1) 
df = pd.read_csv(file_path, skiprows=1)
#skiprows=1: die Überschrift des Datensatzes wird nicht miteingelesen



#Lösche erste und letzte Zeile, da diese fehlende Werte enthält. 
df = df.iloc[1:-1]  # Entfernt die erste und letzte Zeile
df.reset_index(drop=True, inplace=True)  # Setzt den Index neu

print(df.head())

#Informationen über den Datensatz
print(df.info())


#Statistische Zusammenfassung


# Sicherstellen, dass die Werte numerisch sind (falls sie als Strings eingelesen wurden)
df[["J-D", "DJF", "MAM", "JJA", "SON"]] = df[["J-D", "DJF", "MAM", "JJA", "SON"]].apply(pd.to_numeric, errors="coerce")

# Statistische Zusammenfassung berechnen
summary = df[["J-D", "DJF", "MAM", "JJA", "SON"]].describe()

# Ergebnis ausgeben
print(summary)

# Min- & Max-Werte mit zugehörigem Jahr finden
columns = ["J-D", "DJF", "MAM", "JJA", "SON"]
min_values = df.loc[df[columns].idxmin(), ["Year"] + columns]
max_values = df.loc[df[columns].idxmax(), ["Year"] + columns]

# Ergebnis ausgeben
print("Minimum-Werte mit Jahr:")
print(min_values)
print("\nMaximum-Werte mit Jahr:")
print(max_values)





# Zeige die Anzahl der fehlenden Werte pro Spalte
print("Anzahl der fehlenden Werte pro Spalte:")
print(df.isnull().sum())


#Plotte die Veränderung der Temperaturabweichung über alle Jahre 
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["J-D"], marker="o", linestyle="-", label="Jährliche Mitteltemperatur")
plt.xlabel("Jahr")
plt.ylabel("Temperaturabweichung (°C)")
plt.title("Globale Temperaturveränderung über die Jahre")
plt.legend()
plt.grid()
plt.yticks(np.arange(-0.6,max(df["J-D"]) , step=0.2))  # Schrittweite anpassen

plt.show()



print("Wärmste Jahre:")
print(df.nlargest(5, "J-D"))  # Top 5 wärmste Jahre

print("\nKälteste Jahre:")
print(df.nsmallest(5, "J-D"))  # Top 5 kälteste Jahre


#Plotte durchschnittliche Temperaturen für jedes Jahrzehnt
df["Decade"] = df["Year"] -1 - (df["Year"] - 1) % 10  +5# Rundet Jahre auf Jahrzehnte
df_decades = df.groupby("Decade")["J-D"].mean()

plt.figure(figsize=(10, 5))
plt.bar(df_decades.index, df_decades, width=8, color="red", alpha=0.7)
plt.xlabel("Dekade")
plt.ylabel("Mittlere Anomalie (°C)")
plt.title("Mittlere globale Anomalie pro Jahrzehnt")
plt.grid()
plt.show()


#Vergleich Sommer und Winter- Temperaturen 
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["DJF"], marker="o", linestyle="-", label="Winter (DJF)", color="blue")
#plt.plot(df["Year"], df["MAM"], marker="o", label="Frühling (MAM)")
plt.plot(df["Year"], df["JJA"], marker="o", linestyle="-", label="Sommer (JJA)", color="red")
#plt.plot(df["Year"], df["SON"], label="Herbst (SON)")
#r=max(max("DJF"), max("MAM"), max("JJA"), max("SON")) 

plt.xlabel("Jahr")
plt.ylabel("Anomalie (°C)")
plt.title("Vergleich: Temperaturentwicklung im Sommer vs Winter")
plt.legend()
plt.grid()
plt.yticks(np.arange(-0.8, 1.6, step=0.2))  # Schrittweite anpassen
plt.show()




#Polynomiale Regression

########################
#Für alle Daten ab 1880
#######################


#x = df["Year"]
#y = df["J-D"]


#####################
#für Daten ab 1980 
#######################

# Daten filtern 
df_filtered = df[df["Year"] >= 1980]

# Neue x- und y-Werte setzen
x = df_filtered["Year"]
y = df_filtered["J-D"]






# Polynom-Fit (passenden Grad ausprobieren)
p = Polynomial.fit(x, y, 3)  
y_pred = p(x)

# Vorhersage bis 2050
#x_future = np.arange(x.min(), 2051)  # Jahre bis 2050
x_future = np.arange(1980, 2051)  # Jahre von 1980 bis 2050
y_future = p(x_future)  # Temperaturprognose berechnen

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Echte Werte", color="blue", alpha=0.5)
plt.plot(x_future, y_future, label="Vorhersage bis 2050", color="red", linewidth=2, linestyle="dashed")
plt.xlabel("Jahr")
plt.ylabel("Anomalie (°C)")
plt.title("Vorhersage der globalen Temperatur bis 2050 (mit polynomialer Regression)")
plt.legend()
plt.grid()
plt.show()


#Fehler berechnen 
mse_poly = mean_squared_error(y, y_pred)
print(f"Polynomiale MSE: {mse_poly:.4f}")



# Exponentielle Funktion
def exp_func(x, a, b, c):
    #xmin= x.min()    #für Daten von 1880 bis 2024
    xmin = 1980       #für Daten von 1980 bis 2024
    return a * np.exp(b * (x - xmin)) + c

popt, _ = curve_fit(exp_func, x, y, p0=(0.01, 0.01, -0.5))

y_pred2 = exp_func(x, *popt)
y_exp = exp_func(x_future, *popt)

lastelem= y_exp[-1]
print(f"Temperatur 2050 ist: {lastelem}" )

#Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Daten", color="blue", alpha=0.5)
plt.plot(x_future, y_exp, label="Vorhersage bis 2050", color="green", linewidth=2)
plt.xlabel("Jahr")
plt.ylabel("Anomalie (°C)")
plt.title("Vorhersage der globalen Temperatur bis 2050 (mit exponentieller Regression)")
plt.legend()
plt.grid()
plt.show()



#Fehler berechnen für exponentielle Funktion
mse_exp = mean_squared_error(y, y_pred2)
print(f"Exponentielle MSE: {mse_exp:.4f}")







