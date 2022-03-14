# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:04:01 2020

@author: Victor Romero
Nombre: Perceptrón Multicapa
"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

class MLP():
    # constructor
    def __init__(self,xi,d,w_1,w_2, w_3,us,uoc, uoc2,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida, n_hidden_2):
        # Variables de inicialización 
        self.xi = np.transpose(xi)
        self.d = d
        self.w1 = w_1
        self.w2 = w_2
        self.w3 = w_3
        self.us = us # umbral neurona de salida (equivalente a w0)
        self.uoc = uoc# umbrales de las capa oculta 1 (equivalentes a w0)
        self.uoc2 = uoc2 # numbrales en la segunda capa oculta (equivalentes a w0)
        self.precision = precision
        self.epocas = epocas
        self.fac_ap = fac_ap
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_salida = n_salida
        self.n_hidden_2 = n_hidden_2
        # Variables de aprendizaje
        self.di = 0 # Salida deseada en iteracion actual
        self.error_red = 1 # Error total de la red en una conjunto de iteraciones
        self.Ew = 0 # Error cuadratico medio
        self.Error_prev = 0 # Error anterior
        self.Errores = [] # lista de errores del perceptron multuicapa
        self.Error_actual = np.zeros((len(d))) # Errores acumulados en un ciclo de muestras
        self.Entradas = np.zeros((1,n_entradas))
        self.n1 = np.zeros((n_ocultas,1)) # Potencial de activacion en neuronas ocultas (equivalente a n1)
        self.n2 = np.zeros((n_hidden_2,1)) # Potencial de activacion en neuronas ocultas (equivalente a n1)
        self.n3 = 0.0 # Potencial de activacion en neurona de salida

        self.a1 = np.zeros((n_ocultas,1)) # Funcion de activacion de neuronas ocultas (equivalente a a1)
        self.a2 = np.zeros((n_hidden_2,1)) # Funcion de activacion de neuronas ocultas (equivalente a a1)
        self.a3 = 0.0 # Funcion de activacion en neurona de salida
        self.epochs = 0
        # Variables de retropropagacion
        self.error_real = 0
        self.s1 = np.zeros((n_ocultas,1)) # Deltas en neuronas ocultas
        self.s2 = np.zeros((n_hidden_2,1)) # Deltas en neuronas ocultas
        self.s3 = 0.0 # delta de salida
     
    
    def Aprendizaje(self):
        Errores = [] # Almacenar los errores de la red en un ciclo
        while(np.abs(self.error_red) > self.precision):
            self.Error_prev = self.Ew
            for i in range(len(d)):
                self.Entradas = self.xi[:,i] # muestra de los datos
                self.di = self.d[i]
                self.Propagar()
                self.Backpropagation()
                self.Propagar()
                self.Error_actual[i] = (0.5)*((self.di - self.a3)**2)
            # error global de la red
            self.Error()
            Errores.append(self.error_red)
            self.epochs +=1
            # Si se alcanza un mayor numero de epocas
            if self.epochs > self.epocas:
                break
        # Regresar 
        print(np.abs(self.error_red))
        return self.epochs,self.w1,self.w2,self.us,self.uoc,Errores
                
    
    def Propagar(self):
        # Operaciones en la primer capa
        for a in range(self.n_ocultas):
        	# equivalente a n1
            self.n1[a,:] = np.dot(self.w1[a,:], self.Entradas) + self.uoc[a,:]
        
        # Calcular la activacion de la neuronas en la capa oculta
        for o in range(self.n_ocultas):
            self.a1[o,:] = tanh(self.n1[o,:])

        # Operaciones en la primer capa
        for a in range(self.n_hidden_2):
        	# equivalente a n1
            self.n2[a,:] = np.dot(self.w2[a,:], self.a1) + self.uoc2[a,:]

        for o in range(self.n_hidden_2):
            self.a2[o,:] = tanh(self.n2[o,:])
        
        # Calcular el valor de n2 para la neurona de salida
        self.n3 = (np.dot(self.w3,self.a2) + self.us) # n3
        # Calcular la salida de la neurona de salida (equivalente a3)
        self.a3 = tanh(self.n3)
    
    def Backpropagation(self):
        # Calcular el error
        self.error_real = (self.di - self.a3)
        # Calculo de la sensibilidad capa 2
        self.s3 = -2 * (dtanh(self.n3) * self.error_real) # calculo de s3
        # Ajustar w2
        self.w3 = self.w3 + (np.transpose(self.a2) * self.fac_ap * self.s3)
        # Ajustar umbral us (w0)
        self.us = self.us + (self.fac_ap * self.s3)

        # Calcular sensibilidad neuronas de la capa 2
        self.s2 = dtanh(self.n2) * np.transpose(self.w3) * self.s3# calculo de s2
        # Ajustar los pesos w1
        for j in range(self.n_hidden_2):
            self.w2[j,:] = self.w2[j,:] + (self.s2[j,:] * np.transpose(self.a1) * self.fac_ap)
        
        # Ajustar el umbral en las neuronas de la capa oculta numero 2
        for g in range(self.n_hidden_2):
            self.uoc2[g,:] = self.uoc2[g,:] + (self.fac_ap * self.s2[g,:])


        # Calcular sensibilidad neuronas capa oculta numero 1
        self.s1 = dtanh(self.n1) * np.dot(np.transpose(self.w2), self.s2)# calculo de s2
        # Ajustar los pesos w1
        for j in range(self.n_ocultas):
            self.w1[j,:] = self.w1[j,:] + (self.s1[j,:] * np.transpose(self.Entradas) * self.fac_ap)
        
        # Ajustar el umbral en las neuronas de la capa oculta 1
        for g in range(self.n_ocultas):
            self.uoc[g,:] = self.uoc[g,:] + (self.fac_ap * self.s1[g,:])
        
    def Error(self):
        # Error cuadratico medio
        self.Ew = ((1/len(d)) * (sum(self.Error_actual)))
        self.error_red = (self.Ew - self.Error_prev)

# Funcion para obtener la tanh
def tanh(x):
    return np.tanh(x)

# Funcion para obtener la derivada de tanh x
def dtanh(x):
    return 1.0 - np.tanh(x)**2

# Funcion sigmoide de x
def sigmoide(x):
    return 1/(1+np.exp(-x))

# Funcion para obtener la derivada de de la funcion sigmoide
def dsigmoide(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)


# Propagama principal
if "__main__"==__name__:
    xi = np.array([[0.5,1], [2,3], [0,0], [1,1], [-1,0.89], [0.25,1], [0.65,0.45], [-4,-3.98]])
    d = np.array([-1,0,0,1,0,0,-1,1])
    # Parametros de la red
    f,c = xi.shape
    fac_ap = -0.4
    precision = 0.001
    epocas = 10000 #
    epochs = 0
    # Arquitectura de la red
    n_entradas = c # numero de entradas
    cap_ocultas = 1 # Una capa oculta
    n_ocultas = 6 # Neuronas en la capa oculta 1
    n_hidden_2 = 3 # neuronas capa salida
    n_salida = 1 # Neuronas en la capa de salida
    # Valor de umbral o bia
    us = 1.0 # umbral en neurona de salida
    uoc = np.ones((n_ocultas,1),float) # umbral en la capa oculta 1
    uoc2 = np.ones((n_hidden_2, 1), float)
    # Matriz de pesos sinapticos
    random.seed(0) # 
    w_1 = random.rand(n_ocultas,n_entradas)
    w_2 = random.rand(n_hidden_2,n_ocultas)
    w_3 = random.rand(n_salida,n_hidden_2)
    
    #Inicializar la red PMC
    red = MLP(xi,d,w_1,w_2, w_3,us,uoc,uoc2 ,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida, n_hidden_2)
    epochs,w1_a,w2_a,us_a,uoc_a,E = red.Aprendizaje()
    print(epochs)
    
    # graficar el error
    plt.grid()
    plt.ylabel("Error de la red",fontsize=12)
    plt.xlabel("Épocas",fontsize=12)
    plt.title("Perceptrón Multicapa",fontsize=14)
    x = np.arange(epochs)
    plt.plot(x,E,'b',label="Error global")
    plt.legend(loc='upper right')
    plt.show()