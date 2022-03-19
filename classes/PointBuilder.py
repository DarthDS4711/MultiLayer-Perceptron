import numpy as np


class PointBuilder:
    def __init__(self, fig, ax):
        # figuras del canvas
        self.fig = fig
        self.ax = ax
        self.plot = self.ax.scatter([], [], color='red', marker='o')# 0
        self.another = self.ax.scatter([], [], color='blue', marker='o')# 1
        self.class3 = self.ax.scatter([], [], color='green', marker='o')# -1
        # figuras del barrido del canvas
        self.__size_dot = 5
        self.fig_a = self.ax.scatter([], [], color='darkred', marker='.', s=self.__size_dot)
        self.fig_b = self.ax.scatter([], [], color='darkcyan', marker='.', s=self.__size_dot)
        self.fig_c = self.ax.scatter([], [], color='tomato', marker='.', s=self.__size_dot)
        self.fig_d = self.ax.scatter([], [], color='cyan', marker='.', s=self.__size_dot)
        self.fig_e = self.ax.scatter([], [], color='green', marker='.', s=self.__size_dot)
        self.fig_f = self.ax.scatter([], [], color='lime', marker='.', s=self.__size_dot)
        # conexión del evento para detección de clicks
        self.cid = self.fig.figure.canvas.mpl_connect('button_press_event', self)
        # datos de las clases de entrada
        self.dataClass1 = []
        self.dataClass2 = []
        self.dataClass3 = []
        # datos de el barrido
        self.dataPlot1 = []
        self.dataPlot2 = []
        self.dataPlot3 = []
        self.dataPlot4 = []
        self.dataPlot5 = []
        self.dataPlot6 = []
        self.class_data = -1 
        # linea que representa la fontera de decisión
        self.__line, = self.ax.plot(0, 0, 'b-')
        # arreglo de lineas que representan la frontera de desición
        self.__lines = []


    def set_lines_graph(self, n):
        if len(self.__lines) == 0:
            for index in range(n):
                line, = self.ax.plot(0, 0, 'b-')
                self.__lines.append(line)
        else:
            diff = n - len(self.__lines)
            for index in range(diff):
                line, = self.ax.plot(0, 0, 'b-')
                self.__lines.append(line)   
        

    # función que nos actualizará el estado del evento de clicks
    def update_state_event(self, state):
        if state:
            self.fig.figure.canvas.mpl_connect('button_press_event', self)
        else:
            self.fig.canvas.mpl_disconnect(self.cid)

    def __call__(self, event):
        # si la figura no contiene un evento
        if event.inaxes!=self.ax.axes: 
            return
        # si el contador es par se pone de un color diferente que si no lo es
        match self.class_data:
            case 0:
                self.dataClass1.append((event.xdata, event.ydata))
                self.plot.set_offsets(self.dataClass1)
            case 1:
                self.dataClass2.append((event.xdata, event.ydata))
                self.another.set_offsets(self.dataClass2)
            case -1:
                self.dataClass3.append((event.xdata, event.ydata))
                self.class3.set_offsets(self.dataClass3)
        # actualización de la figura
        self.fig.canvas.draw()


    # función que nos actualiza los datos que fueron evaluados, para su ubicación en una clase
    def set_new_points(self, x1, x2, class_data):
        match class_data:
            case 0:
                self.dataPlot3.append((x1, x2))
                self.fig_x.set_offsets(self.dataPlot3)
            case 1:
                self.dataPlot4.append((x1, x2))
                self.fig_y.set_offsets(self.dataPlot4)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    

    # función que nos actualiza la línea de la frontera de decisión
    def update_line(self, weight1, weigth2, theta):
        line_points = np.linspace(-5, 5)
        self.__line.set_xdata(line_points)
        # ecuación de la recta tipo y = mx +b 
        weights_data = (-weight1 * line_points + theta) / weigth2
        self.__line.set_ydata(weights_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_lines(self, data):
        line_points = np.linspace(-1, 1)
        w_1, w_0, n = data
        for index in range(n):
            weigth = w_1[index,:]
            w1 = weigth[0]
            w2 = weigth[1]
            w0 = w_0[index,:]
            self.__lines[index].set_xdata(line_points)
            weights_data = (-w1 * line_points + w0) / w2
            self.__lines[index].set_ydata(weights_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def add_data(self, data_add):
        self.data.append(data_add)

    # metodo que limpia de manera completa, los datos presentes en el programa
    def clear_graph(self, type):
        # limpiar los datos de los arrays para los gráficos
        self.dataPlot4 = []
        self.dataPlot5 = []
        self.dataPlot6 = []
        self.class_data = -2  
        # restablecer los subgraficos del gráfico principal
        self.ax.cla()
        self.plot = self.ax.scatter([], [], color='red', marker='o')
        self.another = self.ax.scatter([], [], color='blue', marker='o')
        self.class3 = self.ax.scatter([], [], color='green', marker='o')
        self.__line, = self.ax.plot(0, 0, 'b-')
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_title('Perceptron Multicapa')
        # reestablecer el barrido del perceptron
        self.fig_a = self.ax.scatter([], [], color='darkred', marker='.', size=self.__size_dot)
        self.fig_b = self.ax.scatter([], [], color='darkcyan', marker='.', size=self.__size_dot)
        self.fig_c = self.ax.scatter([], [], color='tomato', marker='.', size=self.__size_dot)
        self.fig_d = self.ax.scatter([], [], color='cyan', marker='.', size=self.__size_dot)
        self.fig_e = self.ax.scatter([], [], color='green', marker='.', size=self.__size_dot)
        self.fig_f = self.ax.scatter([], [], color='lime', marker='.', size=self.__size_dot)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    # función que dibuja en el plano la superficie de desición adaline
    def draw_desition_mlp_superface(self, mlp):
        line_points = [] # lista de los datos con su respectivas clases predecidas
        n_points = 200
        n_points_y = 20
        feature_x = np.linspace(-1, 1, n_points)
        feature_y = np.linspace(-1, 1, n_points_y)
        for index in range(0, n_points_y):
            y = feature_y[index]
            for subIndex in range(0, n_points):
                x = feature_x[subIndex]
                data = np.array([x, y])
                class_predicted = np.asscalar(np.round(mlp.predict(data)))
                line_points.append((x, y, class_predicted))
        for value in line_points:
            x = value[0]
            y = value[1]
            class_data = value[2]
            self.set_new_points_mlp(x, y, class_data)
        line_points.clear()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

     # función que agrega el barrido al canvas con los datos proporcionados del adaline
    def set_new_points_mlp(self, x1, x2, predict_class):
        if predict_class == 0:
            self.dataPlot1.append((x1, x2))
            self.fig_a.set_offsets(self.dataPlot1)
        elif predict_class == 1:
            self.dataPlot2.append((x1, x2))
            self.fig_b.set_offsets(self.dataPlot2)
        elif predict_class == -1:
            self.dataPlot5.append((x1, x2))
            self.fig_e.set_offsets(self.dataPlot5)
        

        

    def change_class(self, class_data):
        self.class_data = class_data
        print(self.class_data)
    

    def get_data_class_one(self):
        return self.dataClass1
    

    def get_data_class_two(self):
        return self.dataClass2

    def get_data_class_three(self):
        return self.dataClass3
