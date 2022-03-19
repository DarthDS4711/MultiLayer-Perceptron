import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import time


class MultilayerPerceptron:
	def __init__(self):
		self.__xi = np.array([[0,0]])
		self.__d = None
		self.__w1 = None
		self.__w2 = None
		self.__w3 = None
		self.__us = 1.0 # umbral neurona de salida (equivalente a w0)
		self.__uoc = None# umbrales de las capa oculta 1 (equivalentes a w0)
		self.__uoc2 = None # numbrales en la segunda capa oculta (equivalentes a w0)
		self.__precision = 0.00000001
		self.__epocas = None
		self.__fac_ap = 0.2
		self.__n_inputs = 2
		self.__n_hidden1 = 1
		self.__n_exit = 1
		self.__n_hidden2 = 0
		self.__n_layers = 1
        # Variables de aprendizaje
		self.__di = 0 # Salida deseada en iteracion actual
		self.__net_error = 1 # Error total de la red en una conjunto de iteraciones
		self.__Ew = 0 # Error cuadratico medio
		self.__Error_prev = 0 # Error anterior
		self.__Errors = [] # lista de errores del perceptron multuicapa
		self.__Error_actual = None # Errores acumulados en un ciclo de muestras
		self.__Inputs = None
		self.__n1 = None # Potencial de activacion en neuronas ocultas (equivalente a n1)
		self.__n2 = None # Potencial de activacion en neuronas ocultas (equivalente a n1)
		self.__n3 = 0.0 # Potencial de activacion en neurona de salida
		self.__a1 = None # Funcion de activacion de neuronas ocultas (equivalente a a1)
		self.__a2 = None # Funcion de activacion de neuronas ocultas (equivalente a a1)
		self.__a3 = 0.0 # Funcion de activacion en neurona de salida
		self.__epochs = 500
		self.__n_epochs_train = 0
        # Variables de retropropagacion
		self.__real_error = 0
		self.__s1 = None # Deltas en neuronas ocultas
		self.__s2 = None # Deltas en neuronas ocultas
		self.__s3 = 0.0 # delta de salida


	def set_n_hidden_neurons_hidden_layers(self, n_hidden1, n_hidden2):
		# validacion de el numero de capas a procesar
		self.__n_hidden1 = n_hidden1
		if n_hidden2 > 0:
			self.__n_layers = 2
			self.__n_hidden2 = n_hidden2
		else:
			self.__n_layers = 1
			self.__n_hidden2 = 0
		self.__set_data_and_variables()

	def set_random_weigths(self):
		if self.__n_hidden1 > 0:
			self.__w1 = np.random.rand(self.__n_hidden1, self.__n_inputs)
			if self.__n_layers == 2:# caso de dos capas ocultas
				self.__w2 = np.random.rand(self.__n_hidden2, self.__n_hidden1)
				self.__w3 = np.random.rand(self.__n_exit, self.__n_hidden2)
			else:# caso de una única capa oculta
				self.__w3 = np.random.rand(self.__n_exit, self.__n_hidden1)
			return True
		else:
			return False

	def __set_data_and_variables(self):
		# inicialización pesos w0 en la primera capa oculta
		self.__uoc = np.ones((self.__n_hidden1, 1), float)
		# inicialización de nets y sensibilidades
		self.__n1 = np.zeros((self.__n_hidden1,1))
		self.__s1 = np.zeros((self.__n_hidden1,1))
		self.__a1 = np.zeros((self.__n_hidden1,1))
		if self.__n_layers == 2:
			# inicialización matriz pesos w0 capa oculta 2
			self.__uoc2 = np.ones((self.__n_hidden2, 1), float)
			# inicialización de nets y sensibilidades
			self.__n2 = np.zeros((self.__n_hidden2,1))
			self.__s2 = np.zeros((self.__n_hidden2,1))
			self.__a2 = np.zeros((self.__n_hidden2,1))


	# función que regresa los pesos y sus umbrales de la primera capa oculta
	def return_w1(self):
		return self.__w1, self.__uoc, self.__n_hidden1

	def set_data_for_train(self, first_class, second_class, third_class):
		# datos de entrenamiento
		self.__d = np.array([0])
		for data in first_class:# datos clase con un valor deseado de 0
			x1 = data[0]
			x2 = data[1]
			row = np.array([x1,x2])
			self.__xi = np.vstack([self.__xi, row])
			self.__d = np.append(self.__d, 0)
		for data in second_class:# datos clase con un valor deseado de 1
			x1 = data[0]
			x2 = data[1]
			row = np.array([x1,x2])
			self.__xi = np.vstack([self.__xi, row])
			self.__d = np.append(self.__d, 1)
		for data in third_class:# datos clase con un valor deseado de -1
			x1 = data[0]
			x2 = data[1]
			row = np.array([x1,x2])
			self.__xi = np.vstack([self.__xi, row])
			self.__d = np.append(self.__d, -1)

		self.__xi = np.delete(self.__xi, 0, axis=0)
		self.__xi = np.transpose(self.__xi)# transpocisión de los datosd
		self.__d = np.delete(self.__d, 0)# eliminación del dato de inicialización
		self.__Error_actual = np.zeros((len(self.__d)))


	# función principal para el entrenamiento de la red
	def train_net(self, graph_error):
		n_epochs = 0
		done = True
		while(np.abs(self.__net_error) > self.__precision):
			self.__Error_prev = self.__Ew
			for index in range(len(self.__d)):
				self.__Inputs = self.__xi[:,index]
				self.__di = self.__d[index]
				self.__forward()
				self.__backPropagation()
				self.__forward()# detección del nuevo error actual
				self.__Error_actual[index] = (0.5) * ((self.__di - self.__a3)**2)# error actual
				print(f'Deseada: {self.__di},   obtenida: {self.__a3}')

				# actualización de los pesos grafica
			self.__error_mlp()# calculo del error de la red
			print("")
			n_epochs += 1
			graph_error.add_data(np.asscalar(np.abs(self.__net_error)))
			graph_error.update_graph(n_epochs)
			if n_epochs > self.__epochs:
				done = False
				break
		self.__n_epochs_train = n_epochs
		return done


	# función que nos regresa el valor de a3 y n3 en base al numero de capas
	def __set_values_for_n3_a3(self):
		match self.__n_layers:
			case 1:
				self.__n3 = (np.dot(self.__w3,self.__a1) + self.__us) # n3
			case 2:
				self.__n3 = (np.dot(self.__w3,self.__a2) + self.__us) # n3

		self.__a3 = self.__tanh(self.__n3)


	def __forward(self):
		# obtención de las nets de la primera capa oculta
		for i in range(self.__n_hidden1):
			self.__n1[i,:] = np.dot(self.__w1[i,:], self.__Inputs) + self.__uoc[i,:]

		# Calcular la activacion de la neuronas en la capa oculta
		for o in range(self.__n_hidden1):
			self.__a1[o,:] = self.__tanh(self.__n1[o,:])

        # Operaciones en la segunda capa
		for a in range(self.__n_hidden2):
        	# equivalente a n1
			self.__n2[a,:] = np.dot(self.__w2[a,:], self.__a1) + self.__uoc2[a,:]

		for o in range(self.__n_hidden2):
			self.__a2[o,:] = self.__tanh(self.__n2[o,:])


        # calculo de la net para la capa de salida y su salida
		self.__set_values_for_n3_a3()

    # Funcion para obtener la tanh
	def __tanh(self, x):
	    return np.tanh(x)

	# Funcion para obtener la derivada de tanh x
	def __dtanh(self, x):
	    return 1.0 - np.tanh(x)**2

	def __sigmode(self, x):
		return 1 / (1 + np.exp(-x))

	def __dsigmode(self, x):
		return self.__sigmode(x) * (1 - self.__sigmode(x))

	# Función secundaria que modifca los pesos en base a las capas de salida
	def __update_weigths(self):
		match self.__n_layers:
			case 1:
				# caso de una sola capa oculta
				# actualización de los pesos de la capa de salida
				self.__w3 = self.__w3 + (np.transpose(self.__a1) * self.__fac_ap * self.__s3)
				self.__s1 = self.__dtanh(self.__n1) * np.dot(np.transpose(self.__w3), self.__s3)# calculo de s2
			case 2:
				# caso de las dos capas ocultas
				self.__w3 = self.__w3 + (np.transpose(self.__a2) * self.__fac_ap * self.__s3)
				# Calcular sensibilidad neuronas de la capa 2
				self.__s2 = self.__dtanh(self.__n2) * np.transpose(self.__w3) * self.__s3# calculo de s2
		        # Ajustar los pesos w2
				for j in range(self.__n_hidden2):
					self.__w2[j,:] = self.__w2[j,:] + (self.__s2[j,:] * np.transpose(self.__a1) * self.__fac_ap)
		        # Ajustar el umbral en las neuronas de la capa oculta numero 2
				for g in range(self.__n_hidden2):
					self.__uoc2[g,:] = self.__uoc2[g,:] + (self.__fac_ap * self.__s2[g,:])
		        # Calcular sensibilidad neuronas capa oculta numero 1
				self.__s1 = self.__dtanh(self.__n1) * (np.dot(np.transpose(self.__w2), self.__s2))# calculo de s2


    # función principal de backpropagation
	def __backPropagation(self):
		self.__real_error = (self.__di - self.__a3)
    	# calculo de la sensibilidad de la capa de salida
		self.__s3 = (self.__dtanh(self.__n3) * self.__real_error)
    	# Ajustar umbral us (w0)
		self.__us = self.__us + (self.__fac_ap * self.__s3)
		self.__update_weigths()
        # Ajustar los pesos w1
		for j in range(self.__n_hidden1):
			self.__w1[j,:] = self.__w1[j,:] + (self.__s1[j,:] * np.transpose(self.__Inputs) * self.__fac_ap)
        # Ajustar el umbral en las neuronas de la capa oculta 1
		for g in range(self.__n_hidden1):
			self.__uoc[g,:] = self.__uoc[g,:] + (self.__fac_ap * self.__s1[g,:])

    # función para calcular el error de la red
	def __error_mlp(self):
		self.__Ew = ((1 / len(self.__d)) * (sum(self.__Error_actual)))
		self.__net_error = (self.__Ew - self.__Error_prev)


	# función para ingresar el learning rate
	def set_learning_rate(self, learning_rate):
		self.__fac_ap = learning_rate

	# funcón para ingresar el número de épocas maximo
	def set_epochs(self, epochs):
		self.__epochs = epochs

	# función para ingresar el error mínimo deseado
	def set_min_error(self, error):
		self.__precision = error

	# función que predice las clases
	def predict(self, xi):
		xi = np.transpose(xi)
		self.__Inputs = xi
		self.__forward()
		return self.__a3
	

	# función que retorna el error
	def return_error(self):
		return self.__net_error 	

	def return_n_epochs(self):
		return self.__n_epochs_train	
