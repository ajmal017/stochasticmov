# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:17:45 2020

@author: Nicolas
"""

import os
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
os.chdir('C:\\Users\\Nicolas\\Desktop\\UDESA\\Ingenieria Financiera')
import numpy as np

columnas = ["Date", "Close"]
data = pd.read_csv('MELI.csv', sep=",", usecols = columnas).set_index("Date")
data
import yfinance as yf
data = pdr.get_data_yahoo('MELI', start = '2020-01-01', end = '2020-06-01', )
data = data.Close
data = pd.DataFrame(data) #lo transforme en dataframe

###################
##EJERCICIO 1
####################

import matplotlib.pyplot as plt

#Grafico precio MELI normal.

plt.plot(data.Close)
ax = plt.subplot()
ax.set_xticks(range(0,120,20))
ax.set_xticklabels(['1-Enero', '1-Febrero', '1-Marzo', \
                    '1-Abril','1-Mayo','1-Junio'])
plt.ylabel('Precio Cierre')
plt.xlabel('Dias')
plt.title(' Precios Diarios MELI')
plt.show()


#Histograma de precios diarios

plt.hist(data.Close, color = 'Yellow', bins = 15)
plt.ylabel('Frecuencia')
plt.xlabel('Precio Cierre')
plt.title('Histograma Precios Diarios MELI')
plt.show()

#Retorno Log MELI
retlogdata = data.Close.pct_change()
retlogdata = pd.DataFrame(retlogdata, columns = ['Close']) #lo transforme en dataframe
#o tambien puede ser
retlogdata = np.log(data.Close) - np.log(data.Close.shift(1))
retlogdata = retlogdata.dropna()

retlogdata = pd.DataFrame(retlogdata, columns = ['Close']) #lo transforme en dataframe

#drop NA
#Histograma Retorno Logaritmico
plt.hist(retlogdata.Close, color = 'Red', bins = 15)
plt.ylabel('Frecuencia')
plt.xlabel('Precio Cierre')
plt.title('Histograma Retorno Logaritmico Diarios MELI')
plt.show()
#quedo algo medianamente lindo


##EJERCICIO 2##
##Varianza, desvio y media simple

retlogmean = retlogdata.Close.mean()
retlogmean
import numpy as np
np.mean(retlogdata)
#0.00315
retlogmeananual = (1+retlogmean)**360-1
##2.10282022338924
retlogdesvio = np.std(retlogdata)
#0.044862
sigmaanual = retlogdesvio*np.sqrt(360)
#0.85119
np.var(retlogdata.Close)
#0.002012569159142236
varanual = np.var(retlogdata.Close)*360
#0.724524897291205

#EWMA

data2 = data.sort_values(by = 'Date', ascending = True) ##no va
retlogdata['EWM'] = retlogdata.Close.ewm(span=100, adjust=False).std()
#aca me dio la varianza de cada dia, A CHEQUEAR BIEN QUE HICE
#EWMA debe ser el promedio

retlogdataewmastd = retlogdata.EWM.sum()/105
retlogdataewmastd
#0.033909420312743625

retlogdataewmavar = retlogdataewmastd**2
retlogdataewmavar
#0.0014113061919830761

##EJERCICIO 3##
##ACA HAY QUE PONER EWMA
precio_5 = data[103:]*(1+retlogmean)**5
## Close en t = 5
## 857.4594119878731

desvio_5 = [(precio_5 * (1 - retlogdesvio), precio_5 *(1 + retlogdesvio))]
desvio_5                
##  Date             Close     
##  2020-06-01  818.804249,                  
## Date               Close   
## 2020-06-01  896.114575)]

##EJERCICIO 4##
sigma = retlogdataewmastd

import seaborn as sns
S0 = data.Close[103]
delta_t = 1
mu = retlogmean
#precio = =S(i-1)*EXP((mu-0.5*sigma*sigma)*deltat+sigma*SQRT(deltat)*phi'
precio = S0*np.exp((mu-0.5*sigma*sigma)*delta_t+sigma*delta_t**0.5*np.random.normal(0, delta_t, 50))            
precio
np.mean(precio)
#844.1691292232877
##EJERCICIO 5##

from scipy.stats import norm
risk_free = 0.05 #dato comoo TNA que lo pasamos a diario CAMBIARLO
delta_t = 1/90 #esta en dias
S0  #precio hoy
X = S0 #Strike de ejercicio at the money
sigma = retlogdataewmastd*np.sqrt(252)
T = 0.25 #a revisar eesto, anual? semestral? por dia?

#Precio Call

d1 =  (np.log ( S0 / X) + (risk_free + 0.5*(sigma**2)) * T) / (sigma * (T **(1 / 2 )))
d2 = d1 - sigma * T**0.5
Nd1 = norm.cdf(d1)
NNd1 = norm.cdf(-d1)
Nd2 = norm.cdf(d2)
NNd2 = norm.cdf(-d2)
call_price = S0*Nd1-X*np.exp(-risk_free*T)*Nd2
#95.13743143526528


##EJERCICIO 6##
#Precio put por put call parity

put_price = call_price - S0 + X*np.exp(-risk_free*T)
# 84.65210106384723
#Formula directa Put-Call parity comprobado que dan igual
putcallparity = X*np.exp(-risk_free*T)*NNd2-S0*NNd1
# 84.65210106384723 que buenoo


##EJERCICIO 5##
# sensibilidad valor Call cambiando el strike
price_callbs = []
cambios = [770,820,825,830,835,840,844.0800170898438,845,850,855,860,865,870,920,1050,1300]
for i in cambios:
    X_new = i
    d1 =  ( np.log ( S0 / X_new) + (risk_free + 0.5*sigma**2) * T) / (sigma * (T **(1 / 2 )))
    d2 = d1 - sigma * T**0.5
    Nd1 = norm.cdf(d1)
    NNd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(d2)
    NNd2 = norm.cdf(-d2)
    call_price = S0*Nd1-X_new*np.exp(-risk_free*T)*Nd2
    price_callbs.append(call_price)
    price_callbs = pd.DataFrame(price_callbs) #lo transforme en dataframe
    price_callbs['Strike'] = cambios
    price_callbs = pd.DataFrame(price_callbs).set_index('Strike') 
  
 ##falta crear el grafico   
 plt.plot(price_callbs)
 #a priori tiene sentido HACERLO MAS LINDO
 #HACER LO MISMO PARA EL PUT
 
valorintr = pd.DataFrame()
valorintr=[]    
for i in cambios:        
    valorintr =   i - 844.0800170898438
    if (i - 844.0800170898438)>0:
        print (i - 844.0800170898438)
    else:
        print(0)
        
#ESTO DE ARRIBA NO IMPORTA ES PARA OTRO EJERCICIO
        
        
##EJERCICIO 7##
#valor call cambiando precio de la accion        
        
price_callSbs = []
cambios = [600,770,820,825,830,835,840,844.0800170898438,845,850,855,860,865,870,920,1050,1300]
for i in cambios:
    S0_new = i
    d1 =  ( np.log ( S0_new / X) + (risk_free + 0.5*sigma**2) * T) / (sigma * (T **(1 / 2 )))
    d2 = d1 - sigma * T**0.5
    Nd1 = norm.cdf(d1)
    NNd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(d2)
    NNd2 = norm.cdf(-d2)
    call_price = S0_new*Nd1-X*np.exp(-risk_free*T)*Nd2
    price_callSbs.append(call_price)
    price_callSbs = pd.DataFrame(price_callSbs)
    price_callSbs['Precio Accion'] = cambios
    price_callSbs = pd.DataFrame(price_callSbs).set_index('Precio Accion') 
  
valorintr = pd.DataFrame()
valorintr=[]    
for i in cambios:        
    valorintr =   i - 844.0800170898438
    if (i - 844.0800170898438)>0:
        print (i - 844.0800170898438)
    else:
        print(0)
valorintrcall =  [0,0,0,0,0,0,0,0,0.91998291015625,5.91998291015625,10.91998291015625,15.91998291015625,20.91998291015625,25.91998291015625,75.91998291015625,205.91998291015625,455.91998291015625]
price_callSbs['Valor Int'] = valorintrcall    

plt.plot(price_callSbs, valorintrcall)
##valor PUT cambiando valor accion
        
price_putbs = []
cambios = [480,550,600,770,820,825,830,835,840,844.0800170898438,845,850,855,860,865,870,920]
for i in cambios:
    S0_new = i
    d1 =  ( np.log ( S0_new / X) + (risk_free + 0.5*sigma**2) * T) / (sigma * (T **(1 / 2 )))
    d2 = d1 - sigma * T**0.5
    Nd1 = norm.cdf(d1)
    NNd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(d2)
    NNd2 = norm.cdf(-d2)
    call_price = S0_new*Nd1-X*np.exp(-risk_free*T)*Nd2
    put_price = call_price - S0_new + X*np.exp(-risk_free*T)
    price_putbs.append(put_price)
    price_putbs = pd.DataFrame(price_putbs) #lo transforme en dataframe
    price_putbs['Precio Accion'] = cambios
    price_putbs = pd.DataFrame(price_putbs).set_index('Precio Accion') 

valorintr=[]    
for i in cambios:        
    valorintr = 844.0800170898438 - i
    if (844.0800170898438 - i)>0:
        print (844.0800170898438-i)
    else:
        print(0)

valorint =  [364.08001708984375,294.08001708984375,244.08001708984375,74.08001708984375,24.08001708984375,19.08001708984375,14.08001708984375,9.08001708984375,4.08001708984375,0,0,0,0,0,0,0,0]
price_putbs['Valor Int'] = valorint    
plt.figure(figsize=(10,8))
plt.plot(price_putbs)
#se cruzan, hacer el grafico mas lindo pero a apriori todo ok!!!
#ahora si, 
# a revisar bien pero esta ok ver grafico y listo y copiar para los demas puts y calls 
        
        
##EJERCICIO 8##
#5% in the money 5% out of the money
r = 0.05
delta_t = 3/12

#menos 5% el precio de ej de hoy. 
E_menos5 = S0 * 1.05
d18 =  ( np.log ( S0 / E_menos5) + (risk_free + sigma**2/2) * delta_t) / (sigma * delta_t **(1 / 2 ))
d28 = d18 - sigma * np.sqrt(delta_t)
E_PV = E_menos5 * np.exp(-risk_free * delta_t)

call_price = norm.cdf(d18) * S0 - norm.cdf(d28) * E_PV
put_price =  -1 * norm.cdf(-d18) * S0 + norm.cdf(-d28) * E_PV
    
call_in_5 = call_price
put_out_5 = put_price
#mas 5% el precio de ej de hoy. 
E_mas5 = S0 * 0.95
d182 =  ( np.log ( S0 / E_mas5) + (risk_free + var/2) * delta_t) / (desv * delta_t **(1 / 2 ))
d282 = d182 - desv * np.sqrt(delta_t)
E_PV = E_mas5 * np.exp(-risk_free * delta_t)

call_price = norm.cdf(d182) * S0 - norm.cdf(d282) * E_PV
put_price =  -1 * norm.cdf(-d182) * S0 + norm.cdf(-d282) * E_PV

call_out_5 = call_price
put_in_5 = put_price

call_in_5, call_out_5, put_out_5, put_in_5

##EJERCICIO 9## no lo pude resolver aun hoy lo resuelvo
delta_t = 0.25
S0
X
sigma
risk_free
M = 50
###EJERCICO 9## 
def priceeuropeancallMC(S0,X,risk_free2,delta_t,sigma,M):
    np.random.seed(1)
    phi = np.random.randn(M)
    ST = S0*np.exp((risk_free-0.5*sigma**2)*delta_t+sigma*np.sqrt(delta_t)*phi)
    payoff = np.where(ST < X,0,ST - X)
    discountfactor = np.exp(-risk_free*delta_t)
    price_MC = discountfactor*np.mean(payoff)
    return price_MC

price_MC
#58.16413349331626
price_exact = priceeuropeancallMC(S0,X,risk_free,delta_t,sigma,500000)
print("Exact Price:", price_exact)
#con 500 mil simulaciones da un
#Exact Price: 95.24033848763912

stdev_MC = discountfactor*np.std(payoff)/np.sqrt(phi)
##EJERCICIO 10##
##FIJARSE FORMULAS Y CORREGIR PERO ES MASOMENOS SIMPLE, CALCULOS    
##Modelo de Merton
np.random.seed(1)
np.random.randn(10)
T = 0.25
S0
X
sigma #diaria
risk_free2=risk_free
k = 0.02

d1div =  ( np.log ( S0 / X) + (risk_free2-k + 0.5*sigma**2) * T) / (sigma * (T **(1 / 2 )))
d2div = d1 - sigma * T**0.5
Nd1div = norm.cdf(d1)
NNd1div = norm.cdf(-d1)
Nd2div = norm.cdf(d2)
NNd2div = norm.cdf(-d2)
call_pricediv = S0*np.exp(-k*T)*Nd1div-X*np.exp(-risk_free2*T)*Nd2div
put_pricediv = call_pricediv - S0*np.exp(-k*T) + X*np.exp(-risk_free2*T)
putcalldiv = X*np.exp(-risk_free2*T)*NNd2div-S0*np.exp(-k*T)*NNd2div


##EJERCICIO 11##
pi = 3.14159
sigma
T = 0.25
d1
def gammaopteuropeo(S0,sigma,T,d1):
    gamma_c1 = (1/(S0*np.sqrt(2*pi*sigma**2*T)))*np.exp(-d1**2/2)
    return gamma_c1
gamma_c1    
print("Gamma Hedging:", gamma_c1)
gamma_p1 = gamma_c1
#Gamma Hedging: 0.001727510569153466

delta_c1 = Nd1
delta_p1 = delta_c1 - 1

print("Delta Call:", delta_c1)
#Delta Call: 0.5718227835098968
print("Delta Put:", delta_p1)
#Delta Put: -0.4281772164901032

def vegaopteur(S0,T,d1):
    dNd1 = 1/(np.sqrt(2*pi))*np.exp(-(d1**2)/2)
    vega_c1 = S0*np.sqrt(T)*dNd1
    return vega_c1
vega_c1   
print("Vega Option:", vega_c1)
vega_p1 = vega_c1

#Vega Option: 165.63365877451642


##EJERCICIO 12##
#Hay que sacar gamma vega y delta de cada opcion. Sacar el gamma portfolio que es la suma producto de cada opcion y su gamma
#suponemos que el mercado formal comercializa una opcion que tiene un delta 

#Primero calculo delta, gamma y vega para cada opcion
delta_c1,gamma_c1,vega_c1
#(0.5718227835098968, 0.001727510569153466, 165.63365877451642)
delta_p1,gamma_p1,vega_p1
#(-0.4281772164901032, 0.001727510569153466, 165.63365877451642)
delta_c2 = norm.cdf(d182), gamma_c2 = gammaopteuropeo(S0,sigma,T,d182), vega_c2 = vegaopteur(S0,delta_t,d182)
delta_c2,gamma_c2,vega_c2
#((0.6449022392094387,, 0.0016388979876159565, 157.13748725719532)
delta_p2 = delta_c2-1, gamma_p2 = gamma_c2, vega_p2 = vega_c2
delta_p2,gamma_p2,vega_p2
#(-0.3550977607905613, 0.0016388979876159565, 157.13748725719532)


# Delta-gamma neutral
##Agregamos a nuestro portfolio la cantidad Q de unidades de cada subyacente y un call 10% ITM con un vencimiento igual que los otros llamado c3
E_mas15 = S0 * 0.9
d13 =  ( np.log ( S0 / E_mas15) + (risk_free + var/2) * delta_t) / (desv * delta_t **(1 / 2 ))
d23 = d13 - desv * np.sqrt(delta_t)
E_PV = E_mas5 * np.exp(-risk_free * delta_t)
delta_c3 = norm.cdf(d13)
gamma_c3 = gammaopteuropeo(S0,sigma,T,d13)
vega_c3 = vegaopteur(S0,delta_t,d13)
delta_c3,gamma_c3,vega_c3
#(0.7165004376379774, 0.0014906319965916344, 142.9217487235537)


## Establecemos las cantidades de cada opcion
c1 = 2000
c2 = 1000
p1 = 2500
p2 = 500
## Buscamos la cantidad de C3 que nos deja gamma neutral
C3= -(c1*gamma_c1+p1*gamma_p1+c2*gamma_c2+p2*gamma_p2)/(gamma_c3)
print("La cantidad del call 3 que nos deja un gamma neutral es: ", C3)
#La cantidad del call 3 que nos deja un gamma neutral es:  -6864.299549459943

## Una vez que obtenemos C3 encontramos las cantidades de Qc que nos dejen gamma neutral
Qc = -(delta_c1*c1+c2*delta_c2+p1*delta_p1+p2*delta_p2+C3*delta_c3)
print("La cantidad de Qc que nos deja un gamma neutral es: ", Qc)
#La cantidad de Qc que nos deja un gamma neutral es:  4377.717746657527

##Inciso B
A = c1*gamma_c1+p1*gamma_p1+c2*gamma_c2+p2*gamma_p2
B = vega_c1*c1+vega_c2*c2+ vega_p1*p1+vega_p2*p2

# Agregamos  un put llamado P3 al portfolio que es un put out of the money con vencimiento en 90 dias
E_mas15 = S0*0.9
delta_tp3 = 90/360
d1p3 =  ( np.log ( S0 / E_mas15) + (risk_free + var/2) * delta_tp3) / (desv * delta_tp3 **(1 / 2 ))
d2p3 = d1p3 - desv * np.sqrt(delta_tp3)
E_PV = E_menos5 * np.exp(-risk_free * delta_tp3)
delta_p3 = delta_c3-1
gamma_p3 = gamma_c3
vega_p3 = vega_c3
delta_p3,gamma_p3,vega_p3

#(-0.28349956236202256, 0.0014906319965916344, 142.9217487235537)

## Planteamos el sistema de dos ecuaciones que resuelvan el hedge gamma y vega
a = np.array([[gamma_c3,gamma_p3],[vega_c3,vega_p3]])
b = np.array([-A,-B])
Qc,Qp = np.linalg.solve(a,b)
print(Qc,Qp) ##Qc representa cantidades del nuevo Call (C3) y Qp del nuevo Put (P3)
print("Las cantidades del nuevo call y put que hay que comprar para lograr el hedge son:", Qc, Qp)
#Las cantidades del nuevo call y put que hay que comprar para lograr el hedge son: -12018.502269494993 5154.202720035049
## Con estos valores buscamos la cantidad Qc de stock que nos deje delta neutral
Qs = -(delta_c1*c1+delta_c2*c2+delta_p1*p1+delta_p2*p2+delta_c3*Qc+delta_p3*Qp)
print(Qs)
print("Las cantidad de la nueva posicion es de {} Stock, {} de Call 3 y {} de Put 3".format(Qs,Qc,Qp))
#Las cantidad de la nueva posicion es de 9531.920466692576 Stock, -12018.502269494993 de Call 3 y 5154.202720035049 de Put 3

