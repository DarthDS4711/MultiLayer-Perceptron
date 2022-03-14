import numpy as np


class PointBuilder:
    def __init__(self, fig, ax):
        # figuras del canvas
        self.fig = fig
        self.ax = ax
        self.plot = self.ax.scatter([], [], color='red', marker='o')
        self.another = self.ax.scatter([], [], color='blue', marker='o')
        self.class3 = self.ax.scatter([], [], color='green', marker='o')
        # figuras del barrido del canvas
        self.fig_x = self.ax.scatter([], [], color='darkred', marker='.')
        self.fig_y = self.ax.scatter([], [], color='darkcyan', marker='.')
        self.fig_w = self.ax.scatter([], [], color='tomato', marker='.')
        self.fig_z = self.ax.scatter([], [], color='cyan', marker='.')
        # conexión del evento para detección de clicks
        self.cid = self.fig.figure.canvas.mpl_connect('button_press_event', self)
        # datos de las clases de entrada
        self.dataClass1 = []
        self.dataClass2 = []
        self.dataClass3 = []
        # datos de el barrido
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
        for index in range(n):
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
            case 0.5:
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

    # función que agrega el barrido al canvas con los datos proporcionados del adaline
    def set_new_points_adaline(self, x1, x2, predict_class):
        if predict_class >= 0 and predict_class < 0.25:
            self.dataPlot3.append((x1, x2))
            self.fig_x.set_offsets(self.dataPlot3)
        elif predict_class >= 0.25 and predict_class < 0.5:
            self.dataPlot5.append((x1, x2))
            self.fig_w.set_offsets(self.dataPlot5)
        elif predict_class >= 0.5 and predict_class < 0.75:
            self.dataPlot6.append((x1, x2))
            self.fig_z.set_offsets(self.dataPlot6)
        elif predict_class >= 0.75 and predict_class <= 1.0:
            self.dataPlot4.append((x1, x2))
            self.fig_y.set_offsets(self.dataPlot4)

    

    # función que nos actualiza la línea de la frontera de decisión
    def update_line(self, weight1, weigth2, theta):
        line_points = np.linspace(-5, 5)
        self.__line.set_xdata(line_points)
        # ecuación de la recta tipo y = mx +b 
        weights_data = (-weight1 * line_points + theta) / weigth2
        self.__line.set_ydata(weights_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_lines(self, w):
        line_points = np.linspace(-5, 5)
        for index in range(3):
            weigth = w[index]
            w0 = weigth[0]
            w1 = weigth[1]
            w2 = weigth[2]
            self.__lines[index].set_xdata(line_points)
            weights_data = (-w1 * line_points + w0) / w2
            self.__lines[index].set_ydata(weights_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def add_data(self, data_add):
        self.data.append(data_add)

    # metodo que limpia de manera completa, los datos presentes en el programa
    def clear_graph(self):
        # limpiar los datos de los arrays para los gráficos
        self.dataPlot = []
        self.dataPlot2 = []
        self.dataPlot3 = []
        self.dataPlot4 = []
        self.dataPlot5 = []
        self.dataPlot6 = []
        self.class_data = -1  
        # restablecer los subgraficos del gráfico principal
        self.ax.cla()
        self.plot = self.ax.scatter([], [], color='red', marker='o')
        self.another = self.ax.scatter([], [], color='blue', marker='o')
        self.class3 = self.ax.scatter([], [], color='green', marker='o')
        self.__line, = self.ax.plot(0, 0, 'b-')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_title('Perceptron Adaline')
        # reestablecer el barrido del perceptron
        self.fig_x = self.ax.scatter([], [], color='darkred', marker='.')
        self.fig_y = self.ax.scatter([], [], color='darkcyan', marker='.')
        self.fig_w = self.ax.scatter([], [], color='tomato', marker='.')
        self.fig_z = self.ax.scatter([], [], color='cyan', marker='.')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # función que dibuja en el plano la superficie de desición adaline
    def draw_desition_adaline_superface(self, perceptron):
        line_points = [] # lista de los datos con su respectivas clases predecidas
        n_points = 100
        n_points_y = 18
        feature_x = np.linspace(-5, 5, n_points)
        feature_y = np.linspace(-4.7, 4.7, n_points_y)
        for index in range(0, n_points_y):
            y = feature_y[index]
            for subIndex in range(0, n_points):
                x = feature_x[subIndex]
                class_predicted = perceptron.return_value_of_f_y_for_predict(x, y, 1)
                line_points.append((x, y, class_predicted))
        for value in line_points:
            x = value[0]
            y = value[1]
            class_data = value[2]
            self.set_new_points_adaline(x, y, class_data)
        line_points.clear()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        

    def change_class(self, class_data):
        self.class_data = class_data
        print(self.class_data)
    

    def get_data_class_one(self):
        return self.dataPlot
    

    def get_data_class_two(self):
        return self.dataPlot2
