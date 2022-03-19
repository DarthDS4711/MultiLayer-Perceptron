import numpy as np


class GraphSquareError:
	def __init__(self, fig, ax):
		# figura relacionada al grafico
		self.__fig = fig
		self.__ax = ax
		# linea del error
		self.__line, = self.__ax.plot(0, 0, color='red')
		# datos de los errores
		self.__data = []

	def add_data(self, data_add):
		self.__data.append(data_add)

	# funcion que actualiza la grafica del error
	def update_graph(self, epochs):
		line_points = np.linspace(-1, epochs, epochs)
		self.__line.set_xdata(line_points)
		self.__line.set_ydata(np.array(self.__data))
		self.__fig.canvas.draw()
		self.__fig.canvas.flush_events()

	def clear_graph_error(self):
		self.__line, = self.__ax.plot(0, 0, color='red')
		self.__data.clear()
		self.__ax.clear()
		self.__fig.canvas.draw()
		self.__fig.canvas.flush_events()


	def set_number_of_epochs(self, n_epochs):
		self.__ax.set_xlim([-1, (n_epochs + 2)])
		self.__fig.canvas.draw()
		self.__fig.canvas.flush_events()
