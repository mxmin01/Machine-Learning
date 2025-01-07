#Part 3 - Forecasting multivariate time series data using VAR 
import pandas_datareader.data as web
from statsmodels.tsa.api import VAR,adfuller,kpss
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from statsmodels.tools.eval_measures import mse


#Der Reader hat Zugriff auf die FRED API und gibt uns ein pandas Dataframe zurück
#start und Ende der Daten, die gelesen werden sollen 
start = "1990-01-01"
end = "2022-04-01"
economic_df = web.DataReader(["FEDFUNDS", "UNRATE"], "fred", start, end)


#Daten werden analysiert, ob es fehlende Werte oder Null-Werte gibt 
print(economic_df.isna().sum())

#Unsere Daten geben immer den Anfang des Monats an
#Wir passen die Frequenz unseres Dateframes somit an
economic_df = economic_df.asfreq('MS')

#Datensatz wird visuell dargestellt mit Subplots
#Subplot für FEDFUNDS und UNRATE
economic_df.plot(subplots=True)
plt.show()

#Anhand der Darstellung gehen wir davon aus, dass sie sich gegenseitig beeinflussen
#Wenn Fedfunds sinkt, steigt UNRATE

#Beide Variablen müssen stationär sein 
#Mit der check_stationarity Funktion checken wir es ab
def check_stationarity(economic_df):
    kps = kpss(economic_df)
    adf = adfuller(economic_df)
    kpss_pv, adf_pv = kps[1], adf[1]
    
    #ADF-Test prüft die Nullhypothese, dass die Zeitreihe nicht stationär ist
    #Wenn der p-Wert unter 0,05 ist, wird die Nullhypothese verworfen also ist es dann stationär
    
    #KPSS-Test prüft die Nullhypothese, dass die Zeitreihe stationär ist
    #Wenn der p-Wert unter 0,05 ist, wird die Nullhypothese verworfen also ist es dann nicht stationär
    kpssh, adfh = 'Stationary', 'Non-Stationary'
    if adf_pv < 0.05:
        adfh = 'Stationary'
    if kpss_pv < 0.05:
        kpssh = 'Non-Stationary'
    return kpssh, adfh

for i in economic_df.columns:
    kps, adf = check_stationarity(economic_df[i])
    print(f'{i} adf: {adf}, kpss: {kps}')

#beide scheinen stationär zu sein (die eine Abweichung wird ignoriert)




#Der GCS besteht aus 4 Tests zu jedem vergangenen Lags
#Lags ist je nach Kontext Tage, Monate usw. -> 12 Monate
#Anzahl der lags wird mit maxlag festgelegt
#Mit dem Granger casuality test schauen wir nach, ob vergangene Werte einer Variable die Werte der anderen Variable nicht beeinflussen 
#Die Nullhypothese ist, dass die früheren Werte der 2.Variable (FEDFUNDS) die Werte der 1. Variable (UNRATE) nicht beeinflussen
#Vorsicht bei der Reiehnfolge! -> FEDFUNDS MUSS AN 2.Stelle sein
granger = grangercausalitytests(economic_df[['UNRATE', 
'FEDFUNDS']], maxlag=12)

#fast alle lags haben beim Output einen geringeren p-Werte als 0.05
#Das bedeutet, dass wir die Nullhypothese verwerfen können
#Also gibt es einen Einfluss

#ACF und PCF als plot, um rauszufinden ob die Variablen zum AR oder MA Prozess gehören
for col in economic_df.columns:
    fig, ax = plt.subplots(1, 2, figsize=(15, 2))
    plot_acf(economic_df[col], zero=False, lags=30, ax=ax[0], title=f'ACF - {col}')
    plot_pacf(economic_df[col], zero=False, lags=30, ax=ax[1], title=f'PACF - {col}')
    plt.show()
    
#Es ist ein AR    


#Bei Multivariaten Zeitreihen kann eine Skalierung nützlich sein, da die Variablen der Zeitreihen evtl. unterschiedlich sind

#Vor der Skalierung wird der Datensatz in Test und Train getrennt
#train: alles vor 2019, test: alles nach 2020
train = economic_df.loc[:"2019"]
test = economic_df.loc["2020":]
print(f'Train: {len(train)}, Test: {len(test)}')

#Wir importieren Standardscaler zum Skalieren und wenden es mit der Fit-Methode bei unserem Train-Dataset an
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(train)


#Skalierung des Train-Sets
train_sc = pd.DataFrame(scale.transform(train),
                        index=train.index,
                        columns=train.columns)

#Skalierung des Test-Sets
test_sc = pd.DataFrame(scale.transform(test), 
                       index=test.index,
                       columns=test.columns)

#Wir müssen bestimmen wieviele Lags(vorherige Punkte) wir verwenden müssen für die Vorhersage
#Die beste Wahl wird VAR ermitteln -> Ordnung p mit dem geringsten Informationskriterien-Wert
model = VAR(endog=train_sc)
res = model.select_order(maxlags=10)

#Tabelle mit Lags und ihre Informationskriterien-Werten
#geringsten Werte haben ein Stern
print(res.summary())


#Lags mit geringsten Informationskriterienwerten in einer Zeile zusammengefassr
print(res.selected_orders)

#AIC hat den geringsten Wert von allen, wird ausgewählt
results = model.fit(maxlags=7, ic='aic')

#Regressionsergebnisse kann man ignorieren
#Ganz unten ist die Korrelations-Matrix: Diagonale Werte sind 1 -> jede Variable steht in Beziehung mit sich selbst
#Nicht-diagonale Werte -> sehr nah an 0 -> haben fast gar keine lineare Beziehung
#Je näher an 0 desto besser
print(results.summary())

#Unseren Lag speichern wir in lag_order -> verwenden wir später für Vorhersage
lag_order = results.k_ar
print(lag_order)

#Mit forecast-Methode erhalten wir eine Vorhersage inklusive oberer und unterer Konfidenzintervalle
past_y = train_sc[-lag_order:].values
n = test_sc.shape[0]
forecast, lower, upper = results.forecast_interval(past_y, steps=n)

#Visualisierung der Vorhersage inklusive Konfidenzintervalle
idx = test.index
style = 'k--' 

#Trainingsset wird auch dargestellt
ax = train_sc.iloc[:-lag_order, 1].plot(style='k')
pred_forecast = pd.Series(forecast[:, 1], index=idx).plot(ax=ax, style=style)
pred_lower = pd.Series(lower[:, 1], index=idx).plot(ax=ax, style=style)
pred_upper = pd.Series(upper[:, 1], index=idx).plot(ax=ax, style=style)
plt.fill_between(idx, lower[:, 1], upper[:, 1], alpha=0.12)
plt.title('Forecasting Unemployment rate(unrate)')
plt.legend(['Train','Forecast'])
plt.show()

#Grafik zeigt Trainingsdaten und Vorhersage inklusive Konfidenzintervalle
#Wir testen ob ein VAR(7) Modell mit endogenen  Variablen (also das hier) besser ist als ein univartiates AR(7)-Modell
#lag_order ist 7
#Wir nutzen ein ARIMA(7,0,0) 
model = ARIMA(train_sc['UNRATE'], order=(lag_order, 0, 0)).fit()

#Wir visualsieren das ARIMA-Modell
fit = model.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()

#Am Korrelogramm erkennen wir, dass die nötigen Information vom AR-Modell richtig erfasst wurden


#Wir ermitteln die root-mean-squareError Werte der Modelle und erfassen somit welches Modell genauer vorhersagt
#Differenzen zwischen den vorhergesagten Werten eines Modells und den tatsächlichen Werten 
#Je kleiner, desto genauer

#VAR-Ergebnis
print(np.sqrt(mse(test['UNRATE'], forecast[:, 1]))) 

#AR-Ergebnis
print(np.sqrt(mse(test['UNRATE'], model.forecast(n))))
#Beide Modelle performen ähnlich, AR-Modell wird bevorzugt da es simpler ist und geringen Unterschied macht
