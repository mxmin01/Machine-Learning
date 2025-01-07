#Part 1 - Forecasting time series data using auto_arima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.tools import diff
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
plt.rc("figure", figsize=(16, 5))



#pmdarima wird importiert
#Datei wird mit Pfad in milk_file gespeichert und in ein pandas Dataframe eingefügt

#pmdarima wird importiert
import pmdarima as pm

#Pfad für unsere Datei erstellt 
#in milk_file gespeichert und in ein pandas Dataframe eingefügt
milk_file = Path('/Users/Muemin.Kocabas/Desktop/milk_production.csv')
milk = pd.read_csv(milk_file,
                   index_col='month',
                   parse_dates=True)


#train_test_split wird importiert -> stammt von pmdarima
#damit können wir unser Dataset in test und split teilen
#shuffle = false, damit die ursprüngliche Reihenfolge bleibt
#test_size=0.10 -> 10% Testdatensatz (90% Trainingsdatensatz)
from sklearn.model_selection import train_test_split

train, test = train_test_split(milk, test_size=0.10, 
shuffle=False)

#Anzahl der Datenpunkte im Testdatensatz bzw. Trainingsdatensatz
print(f'Train: {train.shape}')
print(f'Test: {test.shape}')

#mit dem train dataset (90%) wird trainiert also das Muster erkannt
#mit dem test dataset (10%) wird getestet, wie gut das Test-Dataset vorhergesagt werden kann

#Um die beste Konfiguration für unser Modell zu finden nutzen wir auto arima
#Zuvor: Excel-Datei analysieren -> Saisonalität erkennbar -> seasonal = True
#m: Anzahl der Perioden innerhalb einer Saison (in unserem Falle Monate)
#test: Unit Root Test testet auf Stationarität, um die Differenzierungsreihenfolge (d) zu bestimmen
#Wir verwenden den ADF-Test (als Standard ist kpss Test festgelegt)
#seasonal_test: Testet auf Saisonalität, um die saisonale Differenzierungsreihenfolge (D) zu finden
#seasonal_test steht nicht im Code, da wir den default-Test OSCB behalten wollen
#infromation_ criterion
#stepwise: schrittweise Suche statt grid search
auto_model = pm.auto_arima(train,
                           
                           #Zuvor: Excel-Datei analysieren -> Saisonalität erkennbar -> seasonal = True
                           seasonal=True,
                           
                           #m: Anzahl der Perioden innerhalb einer Saison (in unserem Falle Monate)
                           m=12,
                           
                           #test: Unit Root Test testet auf Stationarität, um die Differenzierungsreihenfolge (d) zu bestimmen
                           #Wir verwenden den ADF-Test (als Default ist kpss Test festgelegt)
                           test='adf', 
                           
                           #Bewertung der Güte des Modells mit Informationskriterien
                           information_criterion='bic',
                           
                           #stepwise: schrittweise Suche statt grid search
                           stepwise=True,
                           
                           #Die Ergebnisse der Informationskriterien für jedes Modell
                           trace=True)

#Die nicht-saisonalen Reihenfolgen (p,q) und saisonalen Reihenfolgen wurden zuvor mit ACF und PACF Plots erfasst
#auto_arima macht das automatisch



#Durch trace werden uns die Ergebnisse für die Informationskriterien aller Modelle gezeigt
#Bestes Modell wird angezeigt
auto_model.summary()

#Das Modell, welches bei beiden das beste Ergebnis liefert wird gespeichert


#Analyse der allgemeinen Performance des Modells

#Standardisierte Residuen für p: Deseasonalized und Detrend 
#Histogram und QQ-Plot sind nicht perfekt normalverteilt, aber trotzdem ausreichend
#Korrelogramm zeigt keine Autokorrelation
#Es gibt einen sehr hohen Spike bei Lag 0, was normal ist, da dies die Autokorrelation eines Wertes mit sich selbst ist.
#Restliche Werte liegen innerhalb der Konfidenzintervalle
#Indiz dafür, dass das Modell gut ist

auto_model.plot_diagnostics(figsize=(15,7)); plt.show()



#Anzahl der Datensätze im Test-Dataset
n = test.shape[0]

#Mit der Methode predict wird vorhergesagt
#n_periods: Es werden soviele Perioden wie im Test-Dataset vorhergesagt (siehe n)
#return_conf_int: Konfidenzintervalle werden angezeigt
forecast, conf_interval = auto_model.predict(n_periods=n,
return_conf_int=True)

#Visualisierung als Plot
lower_ci, upper_ci  = zip(*conf_interval)
index = test.index
ax = test.plot(style='--', alpha=0.6, figsize=(12,4))
pd.Series(forecast, index=index).plot(style='-', ax=ax)
plt.fill_between(index, lower_ci, upper_ci, alpha=0.2)
plt.legend(['test', 'forecast']); plt.show()

#Test ob Vorhersage innerhalb des Konfidenzintervalls liegt
print(sum(forecast) == sum(conf_interval.mean(axis=1)))
