from tkinter import Entry, Tk, Frame, Button, Label, messagebox, Text, Radiobutton, Spinbox
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from classes.Mutlilayer_Perceptron import MultilayerPerceptron
from classes.PointBuilder import PointBuilder
from validators.validator import *
from classes.GraphError import GraphSquareError


class Window:
    def __init__(self):
        # sección de configuración de la ventana
        self.__window = Tk()
        self.__window.geometry('1280x730')
        self.__window.wm_title('MultilayerPerceptron')
        self.__frame = Frame(self.__window,  bg='gray22', bd=3)
        self.__frame.grid(row=0, column=0, columnspan=10)
        fig, (ax, ax1) = plt.subplots(1, 2)
        fig.set_size_inches(12.5, 4.6)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax1.set_xlim([0, 8000])
        ax1.set_ylim([-0.2, 1])
        ax.set_title('Perceptron Multicapa')
        ax1.set_title('Error cuadrático')
        self.__pointsBuilder = PointBuilder(fig, ax)
        self.__pointsBuilder.set_lines_graph(4)
        self.__graph_error = GraphSquareError(fig, ax1)
        self.__mlp = MultilayerPerceptron()

        # agregar el gráfico a la ventana
        self.__canvas = FigureCanvasTkAgg(fig, master=self.__frame)
        self.__canvas.get_tk_widget().grid(row=0, column=0, columnspan=4)

        # sección de botones del programa
        # botones principales del programa
        self.__btn1 = Button(self.__window, text='Train', width=15, bg='green', fg='white',command=self.train)
        self.__btn2 = Radiobutton(self.__window, text='Red rose', command=self.class_flower_rose, value=1)
        self.__btn3 = Radiobutton(self.__window, text='Blue Rose',command=self.class_flower_blue, value=2)
        self.__btn9 = Radiobutton(self.__window, text='Green Rose',command=self.class_flower_green, value=3)
        self.__btn4 = Button(self.__window, text='Inicialize weigths',
                             command=self.inicialize_random, fg='white', bg='orange')
        
        self.__btn5 = Button(self.__window, width=15, text='Test', command=self.evaluate_points, 
                            state=tkinter.DISABLED, bg='blue', fg='white')
        
        self.__btn6 = Button(self.__window, text='Quit', bg='red', fg='white',command=self.quit, width=15)
        
        self.__btn7 = Button(self.__window, text='Restart', bg='magenta', fg='white',command=self.restart, width=15)
        
        # botones relacionados a los entrys
        self.__btn_epochs = Button(self.__window, text='Asign', command=self.get_epochs, width=15)
        self.__btn_learning_rate = Button(self.__window, text='Asign',command=self.get_learning_rate, width=15)
        self.__btn_min_error = Button(self.__window, text='Asign',command=self.get_min_error, width=15)
        # boton relacionado a los spinbox de las capas ocultas
        self.__btn_hidden = Button(self.__window, text='Asign', width=30, bg='yellow', 
            command=self.get_number_neurons_each_layer)

        # sección de entrys del programa
        # inicialización de entrys
        self.__entry_epochs = Entry(self.__window)
        self.__entry_learning_rate = Entry(self.__window)
        self.__entry_min_error = Entry(self.__window)
         # sección de entrys para la matriz de confusión
        self.__text_true_positives = Text(self.__window, height=1, width=24, state=tkinter.DISABLED)
        self.__text_false_positives = Text(self.__window, height=1, width=24, state=tkinter.DISABLED)
        self.__text_false_negative = Text(self.__window, height=1, width=24, state=tkinter.DISABLED)
        self.__text_true_negative = Text(self.__window, height=1, width=24, state=tkinter.DISABLED)
        # cajas de texto que muestran la información del programa
        self.__text_error = Text(self.__window, height=1, width=24, state=tkinter.DISABLED)
        self.__text_epochs = Text(self.__window, height=1, width=24, state=tkinter.DISABLED)
        self.__text_train = Text(self.__window, height=1, width=24, state=tkinter.DISABLED)

        # sección de los spinbox del programa
        # spinbox relacionados al número de neuronas por capa oculta
        self.__spinbox_hidden_1 = Spinbox(self.__window, from_=1, to=60, wrap=True)# capa oculta 1
        self.__spinbox_hidden_2 = Spinbox(self.__window, from_=0, to=60, wrap=True)# capa oculta 2

        # sección de labels del programa
        # inicialización de labels (relacionados a los entrys)
        Label(self.__window, text='N° Epochs: ').grid(row=3, column=0)
        Label(self.__window, text='Learning Rate: ').grid(row=4, column=0)
        Label(self.__window, text='Min error: ').grid(row=5, column=0)

        # inicialización de label relacionado a entrys de información
        Label(self.__window, text='Final Value theta: ').grid(row=3, column=3)
        Label(self.__window, text='Final number epochs: ').grid(row=4, column=3)
        Label(self.__window, text='Train: ').grid(row=5, column=3)

        # labels relacionados a la matriz de confusión
        # Label(self.__window, text='Positivo').grid(row=6, column=2)
        # Label(self.__window, text='Negativo').grid(row=6, column=3)
        # Label(self.__window, text='Positivo').grid(row=7, column=1)
        # Label(self.__window, text='Negativo').grid(row=8, column=1)

        # labels de resultado de la matriz de confusión
        self.__lbl1 = Label(self.__window, text='0')
        self.__lbl2 = Label(self.__window, text='0')
        self.__lbl3 = Label(self.__window, text='0')
        self.__lbl4 = Label(self.__window, text='0')
        self.__lbl5 = Label(self.__window, text='0')

        # labels relacionados a los spin box de las capas ocultas
        Label(self.__window, text='Número de neuronas por capa oculta').grid(row=4, column=6)
        Label(self.__window, text='Capa oculta 1: ').grid(row=5, column=5)
        Label(self.__window, text='Capa oculta 2: ').grid(row=6, column=5)

        # bloque de variables del programa
        self.__is_initialize_weigths = False
        self.__is_selected_neurons = False


        # sección para inicializar los elementos del programa
        self.set_buttons()
        self.set_entrys()
        # sección para inicializar la matriz de confusión
        # self.set_entry_confuse_matrix()
        self.set_spinbox_hidden_layer()
        self.__window.mainloop()


    def set_spinbox_hidden_layer(self):
        self.__spinbox_hidden_1.grid(row=5, column=6)
        self.__spinbox_hidden_2.grid(row=6, column=6)

    def set_entry_confuse_matrix(self):
        self.__text_true_positives.grid(row=7, column=2)
        self.__text_false_positives.grid(row=7, column=3)
        self.__text_false_negative.grid(row=8, column=2)
        self.__text_true_negative.grid(row=8, column=3)

        # labels de resultado
        self.__lbl1.grid(row=9, column=2) # resultado de verdaderos positivos + falsos positivos
        self.__lbl2.grid(row=9, column=3) # resultado de verdaderos negativos + falsos negativos
        self.__lbl3.grid(row=7, column=4) # resultado de verdaderos positivos + falsos negativos
        self.__lbl4.grid(row=8, column=4) # resultado de verdaderos negativos + falsos positivos
        self.__lbl5.grid(row=9, column=4) # datos totales

    def set_entrys(self):
        # Entrys relacionados a ingresar información al programa
        self.__entry_epochs.grid(row=3, column=1)
        self.__entry_learning_rate.grid(row=4, column=1)
        self.__entry_min_error.grid(row=5, column=1)
        # entrys relacionados con desplegar información
        self.__text_error.grid(row=3, column=4)
        self.__text_epochs.grid(row=4, column=4)
        self.__text_train.grid(row=5, column=4)

    def set_buttons(self):
        # Botones principales del programa
        self.__btn1.grid(row=1, column=0)
        self.__btn2.grid(row=1, column=1)
        self.__btn3.grid(row=1, column=2)
        self.__btn9.grid(row=1, column=3)

        self.__btn4.grid(row=1, column=4)
        self.__btn5.grid(row=1, column=5)
        self.__btn6.grid(row=1, column=6)
        self.__btn7.grid(row=1, column=7)
        # botones relacionados a los entrys
        self.__btn_epochs.grid(row=3, column=2)
        self.__btn_learning_rate.grid(row=4, column=2)
        self.__btn_min_error.grid(row=5, column=2)
        # botones relacionados a los spinbox
        self.__btn_hidden.grid(row=7, column=6)

    # acción del boton para reiniciar la aplicación
    def restart(self):
        # self.__perceptron.restart_perceptron()
        self.__pointsBuilder.clear_graph()
        self.__graph_error.clear_graph_error()
        self.update_buttons_entrys(False)
        self.update_text_boxes(True)
        self.update_content_entrys_and_labels()
        self.__btn5['state'] = tkinter.DISABLED
        self.block_main_buttons(False)
        self.update_text_boxes(False)


    
    # boton para finalizar la execución de la aplicación
    def quit(self):
        self.__window.quit()

    # función que actualiza el estado de las cajas de texto
    def update_text_boxes(self, state):
        if state:
            self.__text_error['state'] = tkinter.NORMAL
            self.__text_epochs['state'] = tkinter.NORMAL
            self.__text_train['state'] = tkinter.NORMAL
            self.__text_true_positives['state'] = tkinter.NORMAL
            self.__text_false_positives['state'] = tkinter.NORMAL
            self.__text_false_negative['state'] = tkinter.NORMAL
            self.__text_true_negative['state'] = tkinter.NORMAL
        else:
            self.__text_error['state'] = tkinter.DISABLED
            self.__text_epochs['state'] = tkinter.DISABLED
            self.__text_train['state'] = tkinter.DISABLED
            self.__text_true_positives['state'] = tkinter.DISABLED
            self.__text_false_positives['state'] = tkinter.DISABLED
            self.__text_false_negative['state'] = tkinter.DISABLED
            self.__text_true_negative['state'] = tkinter.DISABLED

    # función que actualiza los entrys y los label
    def update_content_entrys_and_labels(self):
        # actualización de los entrys
        self.__text_error.delete("1.0", "end")
        self.__text_epochs.delete("1.0", "end")
        self.__text_train.delete("1.0", "end")
        self.__text_true_positives.delete("1.0", "end")
        self.__text_false_positives.delete("1.0", "end")
        self.__text_false_negative.delete("1.0", "end")
        self.__text_true_negative.delete("1.0", "end")

        # actualización de los label
        self.__lbl1["text"] = "0"
        self.__lbl2["text"] = "0" 
        self.__lbl3["text"] = "0"
        self.__lbl4["text"] = "0" 
        self.__lbl5["text"] = "0"

    # función para mostrar mensajes enj pantalla
    def __msg(self, text, title_t):
         messagebox.showinfo(
                message=text, title=title_t)

    # función que obtiene los valores (neuronas) para cada capa oculta
    def get_number_neurons_each_layer(self):
        n_neurons_first_hidden_layer = int(self.__spinbox_hidden_1.get())
        n_neurons_second_hidden_layer = int(self.__spinbox_hidden_2.get())
        self.__mlp.set_n_hidden_neurons_hidden_layers(n_neurons_first_hidden_layer, n_neurons_second_hidden_layer)
        self.__pointsBuilder.set_lines_graph(n_neurons_first_hidden_layer)
        self.__is_selected_neurons = True

    # función que actualiza el estado de los botones relacionados a los inputs
    def update_buttons_entrys(self, state):
        if state:
            self.__btn_epochs['state'] = tkinter.DISABLED
            self.__btn_learning_rate['state'] = tkinter.DISABLED
            self.__btn_min_error['state'] = tkinter.DISABLED
        else:
            self.__btn_epochs['state'] = tkinter.NORMAL
            self.__btn_learning_rate['state'] = tkinter.NORMAL
            self.__btn_min_error['state'] = tkinter.NORMAL


   # función para mostrar la información por pantalla
    def show_info(self, state):
        self.update_text_boxes(True)
        self.__text_error.insert('1.0', str(self.__mlp.return_error()))
        self.__text_epochs.insert('1.0', str(self.__mlp.return_n_epochs()))
        if state:
            self.__text_train.insert('1.0', 'OK')
        else:
            self.__text_train.insert('1.0', 'ERROR')
       
        self.update_text_boxes(False)

    # función que nos valida si existe información previa para entrenar
    def __validate_data_to_train(self):
        number_data_class_one = len(self.__pointsBuilder.get_data_class_one())
        number_data_class_two = len(self.__pointsBuilder.get_data_class_two())
        number_data_class_three = len(self.__pointsBuilder.get_data_class_three())
        if number_data_class_one == 0 or number_data_class_two == 0 or number_data_class_three == 0:
            return False
        else:
            return True


    # metodo que comienza a entrenar con los datos actuales
    def train(self):
        if self.__validate_data_to_train():
            if self.__is_selected_neurons and self.__is_initialize_weigths:
                self.block_main_buttons(True)
                self.__pointsBuilder.update_state_event(False)
                self.update_buttons_entrys(True)
                self.__mlp.set_data_for_train(self.__pointsBuilder.dataClass1, self.__pointsBuilder.dataClass2, 
                    self.__pointsBuilder.dataClass3)

                status = self.__mlp.train_net(self.__graph_error)
                self.show_info(status)

                self.__pointsBuilder.change_class(-2)
                self.__btn5['state'] = tkinter.NORMAL
                self.__btn6['state'] = tkinter.NORMAL
                self.__btn7['state'] = tkinter.NORMAL
                self.__pointsBuilder.update_state_event(True)
            else:
                self.__msg(text='Pesos no iniciados o neuronas no definidas', title_t='Error')
        else:
            self.__msg(text='No existen datos de entrenamiento', title_t='Error')

    # Cambio en el tipo de flor a mapear
    def class_flower_rose(self):
        self.__pointsBuilder.change_class(0)

    def class_flower_blue(self):
        self.__pointsBuilder.change_class(1)

    def class_flower_green(self):
        self.__pointsBuilder.change_class(-1)

    # inicialización de manera aleatoria de los pesos del perceptron
    def inicialize_random(self):
        self.__mlp.set_random_weigths()
        self.__pointsBuilder.update_lines(self.__mlp.return_w1())
        self.__is_initialize_weigths = True
        

    # evaluación de los puntos obtenidos posteriores al entrenamiento
    def evaluate_points(self):
        self.__pointsBuilder.draw_desition_mlp_superface(self.__mlp)

    # función para obtener el learning rate
    def get_learning_rate(self):
        learning_rate = self.__entry_learning_rate.get()
        self.__entry_learning_rate.delete("0", "end")
        if not validate_float(learning_rate):
            messagebox.showinfo(
                message="Factor de aprendizaje incorrecto", title="Error")
        else:
            learning_rate = float(learning_rate)
            self.__mlp.set_learning_rate(learning_rate)
            messagebox.showinfo(
                message="Factor de aprendizaje agregado correctamente", title="Éxito")

    # función que obtiene el error minimo
    def get_min_error(self):
        min_error = self.__entry_min_error.get()
        self.__entry_min_error.delete("0", "end")
        if not validate_float(min_error):
            messagebox.showinfo(
                message="Error minimo incorrecto", title="Error")
        else:
            min_error = float(min_error)
            self.__mlp.set_min_error(min_error)
            messagebox.showinfo(
                message="Error minimo agregado correctamente", title="Éxito")


    # función relacionada a la obtención del número de epocas
    def get_epochs(self):
        n_epochs = self.__entry_epochs.get()
        self.__entry_epochs.delete("0", "end")
        if not validate_integer(n_epochs):
            messagebox.showinfo(
                message="Número de epocas incorrecto", title="Error")
        else:
            n_epochs = int(n_epochs)
            self.__mlp.set_epochs(n_epochs)
            self.__graph_error.set_number_of_epochs(n_epochs)
            messagebox.showinfo(
                message="Número de epocas agregado correctamente", title="Éxito")

    def block_main_buttons(self, state):
        if state:
            self.__btn1['state'] = tkinter.DISABLED
            self.__btn2['state'] = tkinter.DISABLED
            self.__btn3['state'] = tkinter.DISABLED
            self.__btn4['state'] = tkinter.DISABLED
            self.__btn6['state'] = tkinter.DISABLED
            self.__btn7['state'] = tkinter.DISABLED
        else:
            self.__btn1['state'] = tkinter.NORMAL
            self.__btn2['state'] = tkinter.NORMAL
            self.__btn3['state'] = tkinter.NORMAL
            self.__btn4['state'] = tkinter.NORMAL
            self.__btn6['state'] = tkinter.NORMAL
            self.__btn7['state'] = tkinter.NORMAL

