import numpy as np


class GraphSquareError:
	def __init__(self, fig, ax):
		# figura relacionada al grafico
		self.__fig = fig
		self.__ax = ax
		# linea del error
		self.__line = self.__ax.plot(0, 0, 'b-')
		# datos de los errores
		self.__data = []

	def add_data(self, data_add):
		self.__data.append(data_add)

	# funcion que actualiza la grafica del error
	def update_graph(self, epochs):
		line_points = np.arange(epochs)
		self.__ax.plot(line_points, self.__data)
		self.__fig.canvas.draw()
		self.__fig.canvas.flush_events()

	def clear_graph_error(self):
		self.__line = self.__ax.plot(0, 0, 'b-')
		self.__data.clear()
		self.__ax.clear()
		self.__fig.canvas.draw()
		self.__fig.canvas.flush_events()
