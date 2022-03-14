import time
import numpy as np
import math

class Perceptron:
    def __init__(self):
        # pesos del perceptron
        self.__weigth0 = 0
        self.__weigth1 = 0
        self.__weigth2 = 0
        # datos entrada
        self.__x1 = []
        self.__x2 = []
        # salida esperada
        self.__y = []
        # variable theta
        self.__theta = 0
        # variable de factor de aprendizaje
        self.__factor_learning = 0
        # variable de número de épocas máximo
        self.__epochs = 0
        # variable bias
        self.__bias = -1
        # variable de estado del perceptron
        self.__done_learn = False
        # variable de numero de épocas requeridas
        self.__number_of_epochs = 0
        # variable que representa la presión del adaline
        self.__precision = 0

    def restart_perceptron(self):
        self.__weigth0 = 0
        self.__weigth1 = 0
        self.__weigth2 = 0
        self.__x1 = []
        self.__x2 = []
        self.__y = []
        self.__theta = 0
        self.__factor_learning = 0
        self.__epochs = 0
        self.__bias = -1
        self.__done_learn = False
        self.__number_of_epochs = 0
        self.__precision = 0


    def get_epochs(self):
        return self.__epochs

    def get_factor_learning(self):
        return self.__factor_learning
    

    def get_number_of_epochs(self):
        return self.__number_of_epochs


    def get_status_perceptron(self):
        return self.__done_learn

    # función que nos devuelve el valor de la net, evaluado por la función de activación
    def __return_value_of_z(self, index):
        z = (self.__x1[index] * self.__weigth1) + \
            (self.__x2[index] * self.__weigth2) +  (self.__weigth0 * self.__bias)
        if z >= 0:
            return 1
        else:
            return 0

    # función que nos devuelve el valor de clase predecida para un nuevo elemento
    def return_value_of_z_out_of_train(self, x1, x2):
        z = (x1 * self.__weigth1) + (x2 * self.__weigth2) + (self.__weigth0 * self.__bias)
        if z >= 0:
            return 1
        else:
            return 0

    def return_value_of_f_y_for_predict(self, x1, x2, type):
        y = (x1 * self.__weigth1) + (x2 * self.__weigth2) +  (self.__weigth0 * self.__bias)
        y = self.__sigmode(y)
        match type:
            case 1:
                return y
            case 2:
                return round(y)

    # funcion que nos hace el la obtención de los datos
    def __obtain_data(self, pointBuilder):
        data = []
        for value in pointBuilder.dataPlot:
            data.append((value[0], value[1]))
        pointBuilder.dataPlot = []
        for value in pointBuilder.dataPlot2:
            data.append((value[0], value[1]))
        pointBuilder.dataPlot2 = []
        return data  

    # función de predicción de los nuevos datos
    def predict_data(self, pointBuilder):
        pass

    def get_weigth1(self):
        return self.__weigth1
    
    def get_weigth2(self):
        return self.__weigth2
    

    def get_theta(self):
        return self.__weigth0


    # ajuste de los pesos conforme al indice donde se encontró el error 
    def __adjust_weigths(self, error, index):
        self.__weigth1 = self.__weigth1 + (self.__x1[index] * error * self.__factor_learning)
        self.__weigth2 = self.__weigth2 + (self.__x2[index] * error * self.__factor_learning)
        self.__weigth0 = self.__weigth0 + (self.__bias * error * self.__factor_learning)

    # función entrenamiento sin adaline
    def train(self, pointBuilder):
        done = False
        n_epochs = 0
        while (not done and n_epochs < self.__epochs):
            done = True
            for index in range(0, len(self.__y)):
                z = self.__return_value_of_z(index)
                if z != self.__y[index]:
                    done = False
                    # calcular el error
                    error = (self.__y[index] - z)
                    # ajuste del valor de thetha
                    self.__adjust_weigths(error, index)
                    n_epochs += 1
                    pointBuilder.update_line(self.__weigth1, self.__weigth2, self.__weigth0)
                    time.sleep(1)
        if n_epochs < self.__epochs:
            self.__done_learn = True
        self.__number_of_epochs = n_epochs
        pointBuilder.update_line(self.__weigth1, self.__weigth2, self.__weigth0)


    # función sigmode
    def __sigmode(self, y):
        return (1 / (1 + math.e ** -y))

    # función que devuelve el valor de la net 
    def __return_value_of_net(self, index):
        y = (self.__x1[index] * self.__weigth1) + \
            (self.__x2[index] * self.__weigth2) +  (self.__weigth0 * self.__bias)
        return self.__sigmode(y)


     # ajuste de pesos conforme a Adaline
    def __adjust_weigths_adaline(self, error, index, f_y):
        self.__weigth1 = self.__weigth1 + (self.__factor_learning * error * (f_y * (1 - f_y)) * self.__x1[index])
        self.__weigth2 = self.__weigth2 + (self.__factor_learning * error * (f_y * (1 - f_y)) * self.__x2[index])
        self.__weigth0 = self.__weigth0 + (self.__factor_learning * error * (f_y * (1 - f_y)) * self.__bias)


    # función de entrenamiento con adaline
    def train_adaline(self, pointBuilder, graph_error, type_train):
        E_actual = 0
        error = 1
        error_w = 0 # error cuadrático medio
        error_total = 0
        error_prev = 0
        n_samples = len(self.__y)
        n_epochs = 0
        if type_train == 'comparative':
            self.inicialize_weigths()
        while (n_epochs < self.__epochs) and (np.abs(error) > self.__precision):
            error_prev = error_w
            for index in range (0, n_samples):
                y = self.__return_value_of_net(index)
                E_actual = (self.__y[index] - y)
                self.__adjust_weigths_adaline(E_actual, index, y)
                error_total = error_total + ((E_actual) ** 2)
                if type_train == 'non-comparative':
                    pointBuilder.update_line(self.__weigth1, self.__weigth2, self.__weigth0)
                    time.sleep(0.1)
            # calcular error cuadratico medio
            error_w = ((1 / n_samples) * (error_total))
            error = (error_w - error_prev)
            n_epochs += 1
            graph_error.add_data(error)
            graph_error.update_graph(n_epochs)
                
        if n_epochs < self.__epochs:
            self.__done_learn = True
        self.__number_of_epochs = n_epochs

    # subfunciones para la evaluación de casos
    def __positive_case(self, x1, x2):
        y = self.return_value_of_f_y_for_predict(x1, x2, 2)
        print(y)
        return True if y == 0 else False

    def __negative_case(self, x1, x2):
        y = self.return_value_of_f_y_for_predict(x1, x2, 2)
        print(y)
        return True if y == 1 else False

    # función que nos devuelve los valores de la matriz de confusión
    def return_data_of_confuse_matrix(self):
        n_samples = len(self.__y)
        # variables a retornar
        n_true_positive = 0
        n_true_negative = 0
        n_false_positive = 0
        n_false_negative = 0
        for index in range(0, n_samples):
            x1 = self.__x1[index]
            x2 = self.__x2[index]
            match self.__y[index]:
                case 0:
                    if self.__positive_case(x1, x2):
                        n_true_positive += 1
                    else:
                        n_false_positive += 1
                case 1:
                    if self.__negative_case(x1, x2):
                        n_true_negative += 1
                    else:
                        n_false_negative += 1
        return n_true_positive, n_false_positive, n_true_negative, n_false_negative

        
    def set_min_error(self, min_error):
        self.__precision = min_error


    def set_epochs(self, epochs):
        self.__epochs = epochs
    
    def set_factor_learning(self, factor_learning):
        self.__factor_learning = 2 * factor_learning

    def return_n_samples(self):
        return len(self.__y)

    
    def set_inputs_outpus(self, xdata, xdata1):
        for data in xdata:
            self.__x1.append(data[0])
            self.__x2.append(data[1])
            self.__y.append(0)
        for data in xdata1:
            self.__x1.append(data[0])
            self.__x2.append(data[1])
            self.__y.append(1)
        
    def inicialize_weigths(self):
        self.__weigth0 = np.random.uniform(0, 1)
        self.__weigth1 = np.random.uniform(0, 1)
        self.__weigth2 = np.random.uniform(0, 1)
