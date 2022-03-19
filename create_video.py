import time

import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import tensorflow as tf
from matplotlib import animation
from scipy.integrate import solve_ivp

g = 9.8
args_of_pend = {'l': 2, 'm': 1, 'g': 9.8}


class HNN(tf.keras.Model):
    """
    HNN for single pendulum. It have one hidden layer. When creating hnn model,
    you must give a list of input shapes and hidden shapes.
    """

    def __init__(self, shapes, activation='sigmoid'):
        super(HNN, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(shapes[0])
        self.hidden_layer = tf.keras.layers.Dense(shapes[1], activation=activation)
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')  # Один выходной слой, мы создаем гамильтониан
        self._set_inputs(tf.TensorSpec([None, shapes[0]], tf.float32, 'inputs'))
        self.shapes = shapes
        self.activation = activation
        self.W = tf.constant(np.concatenate([np.eye(shapes[0])[shapes[0] // 2:],
                                             -np.eye(shapes[0])[:shapes[0] // 2]], axis=0))

        self.history_loss = []

    def call(self, x):
        return self.output_layer(self.hidden_layer(x))

    def forward(self, x):
        with tf.GradientTape() as tape:
            y = self.hidden_layer(x)
            y = self.output_layer(y)

        y = tape.gradient(y, x)
        return y @ self.W

    def fit(self, x, y, learning_rate=1e-4, epochs=100, verbose=10):
        loss_obj = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for i in range(epochs):

            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                predictions = self.forward(tf.Variable(x))
                loss = loss_obj(tf.Variable(y), predictions)

            grads = tape.gradient(loss, self.trainable_variables, unconnected_gradients='zero')
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.history_loss.append(loss.numpy())

            if verbose > 0:
                if i % verbose == 0:
                    print(f' Step {i}: MSE: {loss.numpy() : 0.6f}')

        print(f'Last train Loss: {loss.numpy() : 0.6f}')

        return self.history_loss

    def predict(self, t_span, y0, **kwargs):

        def s_h(t, x):
            x = tf.Variable(tf.reshape(x, (1, self.shapes[0])), dtype='double')
            return self.forward(x)

        return solve_ivp(s_h, t_span, y0, **kwargs)['y']

    def save(self, path):
        Y, b, alpha = self.weights[:3]

        with open(path, 'wb') as f:
            np.save(f, np.array(self.activation))
            np.save(f, Y.numpy())
            np.save(f, b.numpy())
            np.save(f, alpha.numpy())

        return

    @classmethod
    def load(cls, path):
        '''
        You must get the dimensions of the input and hidden layer
        '''
        with open(path, 'rb') as f:
            activation = np.load(f)
            Y = np.load(f)
            b = np.load(f)
            alpha = np.load(f)

        model = cls(Y.shape, str(activation))
        model.call(tf.constant([[0, 0]]))
        model.hidden_layer.set_weights([Y, b])
        model.output_layer.set_weights([alpha, np.array([0.])])

        return model


class HNND(tf.keras.Model):
    """
    HNN for single pendulum. It have one hidden layer. When creating hnn model,
    you must give a list of input shapes and hidden shapes.
    """

    def __init__(self, shapes, activation='sigmoid'):
        super(HNND, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(shapes[0])
        self.hidden_layer_1 = tf.keras.layers.Dense(shapes[1], activation=activation)
        self.hidden_layer_2 = tf.keras.layers.Dense(shapes[2], activation=activation)
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')  # Один выходной слой, мы создаем гамильтониан
        self._set_inputs(tf.TensorSpec([None, shapes[0]], tf.float32, 'inputs'))
        self.shapes = shapes
        self.activation = activation
        self.W = tf.constant(np.concatenate([np.eye(shapes[0])[shapes[0] // 2:],
                                             -np.eye(shapes[0])[:shapes[0] // 2]], axis=0))

        self.history_loss = []

    def call(self, x):
        x = self.hidden_layer_1(x)
        return self.output_layer(self.hidden_layer_2(x))

    def forward(self, x):
        with tf.GradientTape() as tape:
            y = self.hidden_layer_1(x)
            y = self.hidden_layer_2(y)
            y = self.output_layer(y)

        y = tape.gradient(y, x)
        return y @ self.W

    def fit(self, x, y, learning_rate=1e-4, epochs=100, verbose=10):
        loss_obj = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for i in range(epochs):

            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                predictions = self.forward(tf.Variable(x))
                loss = loss_obj(tf.Variable(y), predictions)

            grads = tape.gradient(loss, self.trainable_variables, unconnected_gradients='zero')
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.history_loss.append(loss.numpy())

            if verbose > 0:
                if i % verbose == 0:
                    print(f' Step {i}: MSE: {loss.numpy() : 0.6f}')

        print(f'Last train Loss: {loss.numpy() : 0.6f}')

        return self.history_loss

    def predict(self, t_span, y0, **kwargs):

        def s_h(t, x):
            x = tf.Variable(tf.reshape(x, (1, self.shapes[0])), dtype='double')
            return self.forward(x)

        return solve_ivp(s_h, t_span, y0, **kwargs)['y']

    def save(self, path):
        Y_0, b_0, Y_1, b_1, alpha = self.weights[:5]

        with open(path, 'wb') as f:
            np.save(f, np.array(self.activation))
            np.save(f, Y_0.numpy())
            np.save(f, b_0.numpy())
            np.save(f, Y_1.numpy())
            np.save(f, b_1.numpy())
            np.save(f, alpha.numpy())

        return

    @classmethod
    def load(cls, path):
        '''
        You must get the dimensions of the input and hidden layer
        '''
        with open(path, 'rb') as f:
            activation = np.load(f)
            Y_0 = np.load(f)
            b_0 = np.load(f)
            Y_1 = np.load(f)
            b_1 = np.load(f)
            alpha = np.load(f)

        model = cls([Y_0.shape[0], Y_0.shape[1], Y_1.shape[1]], str(activation))
        model.call(tf.constant([[0 for _ in range(Y_0.shape[0])]]))
        model.hidden_layer_1.set_weights([Y_0, b_0])
        model.hidden_layer_2.set_weights([Y_1, b_1])
        model.output_layer.set_weights([alpha, np.array([0.])])

        return model


def H(q, p, l, m, g):
    return (p ** 2 / (2 * m * l ** 2)) + (m * g * l * (1 - np.cos(q)))


def S(t, pq=(1, 1), H=H, NN=False, h=1e-7):
    if NN:
        dqdt = (H(np.array([[pq[0], pq[1] + h]])) - H(np.array([[pq[0], pq[1] - h]]))) / (2 * h)
        dpdt = -(H(np.array([[pq[0] + h, pq[1]]])) - H(np.array([[pq[0] - h, pq[1]]]))) / (2 * h)

    else:
        dqdt = (H(pq[0], pq[1] + h, **args_of_pend) - H(pq[0], pq[1] - h, **args_of_pend)) / (2 * h)
        dpdt = -(H(pq[0] + h, pq[1], **args_of_pend) - H(pq[0] - h, pq[1], **args_of_pend)) / (2 * h)

    S = np.array([dqdt, dpdt])

    return S


def give_data(size=9, frames=72, qstarts=(-0.4, 0., 0.4), pstarts=(1., 1., 1.), length_pend=2):
    time_end = frames / 24
    packs = []
    params = tf.random.shuffle(np.array(np.meshgrid(qstarts, pstarts)).reshape(
        (2, len(qstarts) * len(pstarts))).T)  # Декартово произведение всех состояний

    params = params[:size]
    for i in range(size):
        q, p = solve_ivp(fun=S,
                         t_span=[0, time_end],
                         method='RK45',
                         y0=params[i],
                         t_eval=np.linspace(0, time_end, frames, endpoint=True),
                         args=(H, False, 0.1))['y']

        dH = []
        for j in range(frames):
            dH.append(S(0, (q[j], p[j])))

        dH = np.array(dH)
        packs.append(np.reshape(np.concatenate((q, p, dH[:, 0], dH[:, 1]), axis=0), (4, frames)).T)

    return np.array(packs)


def Create_Plot_with_predict(path, x, y, x_HNN, y_HNN, frames=72, q=0, p=0, name='Two pendulums predicted and real'):
    L = 1.5
    # Создаем временную шкалу
    fps = 24  # Количество кадров в секунду
    TimeFinal = frames / fps  # Количество секунд видео
    TimeAmount = int(TimeFinal * fps)
    AxisTime = np.linspace(0, TimeFinal, TimeAmount)
    dt = AxisTime[1] - AxisTime[0]

    # Создадим фигуру, на которой будет происходить отрисовка
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlim(-L - 1, L + 1), plt.ylim(-L - 1, L + 1)
    Pendulum, = ax.plot([], [], 'o-', markersize=20, lw=2, markevery=100000, c='red', alpha=0.5,
                        label='True pendulum')
    Pendulum_line, = ax.plot([], [], c='red')
    Pendulum_HNN, = ax.plot([], [], 'o-', markersize=20, lw=2, markevery=100000, c='navy', alpha=0.5,
                            label='Predicted pendlum')
    Pendulum_HNN_line, = ax.plot([], [], c='navy')

    CenterDot, = ax.plot([], [], 'o', markersize=7, c='grey')
    TimeOnPlot = ax.text(0.05, 0.9, '', transform=ax.transAxes,
                         fontsize=16)  # Время будет отрисовываться в точке 0.05 по оХ и 0.9

    # по оY относительно всего графика

    ax.get_xaxis().set_ticks([]), ax.get_yaxis().set_ticks([])
    ax.set_title(name + f'\nq = {q}, p = {p}', fontsize=22)
    plt.legend(fontsize=22)
    plt.close('all')

    def animate(i):  # Функция, которая на i-ом шаге отрисовывает данные
        Pendulum_HNN.set_data(x_HNN[i:i - 5:-1], y_HNN[i:i - 5:-1])
        Pendulum.set_data(x[i:i - 5:-1], y[i:i - 5:-1])  # Сам маятник
        if i >= 5:
            Pendulum_line.set_data([0, x[i]], [0, y[i]])
            Pendulum_HNN_line.set_data([0, x_HNN[i]], [0, y_HNN[i]])
        CenterDot.set_data([0, 0], [0, 0])  # Подвес

        TimeOnPlot.set_text(f'Time = {i * dt : 0.3f}s')  # Отрисовываем время
        return Pendulum_HNN, Pendulum, CenterDot, Pendulum_line, Pendulum_HNN_line, TimeOnPlot

    anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=0.5)

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    # anim.save(f, writer = writer)
    writergif = animation.PillowWriter(fps=fps, bitrate=800)
    anim.save(path + '.gif', writer=writergif)
    clip = mp.VideoFileClip(f"{path}.gif")
    clip.write_videofile(f"{path}.mp4")


def give_video_sp(q, p, path):
    start_time = time.time()
    frame = 360
    x_test = give_data(size=1, frames=frame, qstarts=[q], pstarts=[p])[:, :, :2]
    x = 2 * np.sin(x_test[0, :, 0])
    y = 2 * -np.cos(x_test[0, :, 0])

    q_, p_ = hnn.predict([0, frame / 24],
                         (x_test[0, 0, :]),
                         t_eval=np.linspace(0, frame / 24, frame, endpoint=True))

    x_pred = 2 * np.sin(q_)
    y_pred = 2 * -np.cos(q_)

    time_calc = time.time() - start_time
    start_time = time.time()
    Create_Plot_with_predict(path, x, y, x_pred, y_pred, frame, q, p)

    return path, time_calc, time.time() - start_time


def create_film(path, x, y, x_hnn=None, y_hnn=None, frame=72,
                names=('True', 'predict by HNN'), fps=24, initial_state=[0, 0, 0, 0]):
    # Создаем временную шкалу
    TimeFinal = frame / fps  # Количество секунд видео
    TimeAmount = int(TimeFinal * fps)
    AxisTime = np.linspace(0, TimeFinal, TimeAmount)
    dt = AxisTime[1] - AxisTime[0]
    # Создадим фигуру, на которой будет происходить отрисовка
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlim(-2.2, 2.2), plt.ylim(-2.2, 2.2)

    FirstPendulum, = ax.plot([], [], 'o-', markersize=16, lw=2, alpha=0.7, markevery=100000, c='navy', label=names[0])
    FirstLine, = ax.plot([], [], '-', lw=3, c='cornflowerblue')
    SecondPendulum, = ax.plot([], [], 'o-', markersize=16, lw=2, alpha=0.7, markevery=100000, c='navy')
    SecondLine, = ax.plot([], [], '-', lw=3, c='cornflowerblue')
    CenterDot, = ax.plot([], [], 'o', markersize=12, c='navy', alpha=0.8)
    TimeOnPlot = ax.text(0.05, 0.9, '', transform=ax.transAxes,
                         fontsize=22)  # Время будет отрисовываться в точке 0.05 по оХ и 0.9
    # по оY относительно всего графика

    FirstPendulum_hnn, = ax.plot([], [], 'o-', markersize=16, lw=2, alpha=0.7, markevery=100000, c='darkred',
                                 label=names[1])
    FirstLine_hnn, = ax.plot([], [], '-', lw=3, c='lightcoral')
    SecondPendulum_hnn, = ax.plot([], [], 'o-', markersize=16, lw=2, alpha=0.7, markevery=100000, c='darkred')
    SecondLine_hnn, = ax.plot([], [], '-', lw=3, c='lightcoral')

    ax.get_xaxis().set_ticks([]), ax.get_yaxis().set_ticks([])
    ax.set_title(f'Two double pendulums. \nInitial state: {initial_state}', fontsize=22), ax.legend(fontsize=22)
    plt.close('all')

    def animate(i):  # Функция, которая на i-ом шаге отрисовывает данные
        if i <= 5:
            j = 0
        else:
            j = i - 5
        FirstLine.set_data([0, x[0, i]], [0, y[0, i]])  # Линия, соединяющая подвес и маятник
        FirstPendulum.set_data(x[0, i], y[0, i])  # Сам маятник
        SecondLine.set_data([x[0, i], x[1, i]], [y[0, i], y[1, i]])
        SecondPendulum.set_data(x[1, i:j:-1], y[1, i:j:-1])

        FirstLine_hnn.set_data([0, x_hnn[0, i]], [0, y_hnn[0, i]])  # Линия, соединяющая подвес и маятник
        FirstPendulum_hnn.set_data(x_hnn[0, i], y_hnn[0, i])  # Сам маятник
        SecondLine_hnn.set_data([x_hnn[0, i], x_hnn[1, i]], [y_hnn[0, i], y_hnn[1, i]])
        SecondPendulum_hnn.set_data(x_hnn[1, i:j:-1], y_hnn[1, i:j:-1])

        CenterDot.set_data([0, 0], [0, 0])  # Подвес

        TimeOnPlot.set_text(f'Time = {i * dt : 0.3f}s')  # Отрисовываем время
        return FirstPendulum_hnn, SecondPendulum_hnn, FirstPendulum, SecondPendulum, CenterDot, TimeOnPlot

    anim = animation.FuncAnimation(fig, animate, frames=TimeAmount)

    writergif = animation.PillowWriter(fps=fps, bitrate=800)
    anim.save(path + '.gif', writer=writergif)
    clip = mp.VideoFileClip(f"{path}.gif")
    clip.write_videofile(f"{path}.mp4")
    return path


def S_dp(t, qp=(-1, -1, 5, 5), h=1e-7):
    def H(q1, q2, p1, p2):
        T = (p1 ** 2 - 2 * p1 * p2 * np.cos(q1 - q2) + 2 * p2 ** 2) / (2 + 2 * np.sin(q1 - q2) ** 2)
        V = -2 * g * np.cos(q1) - g * np.cos(q2)
        return T + V

    farg = np.array([qp[0] + h, qp[1], qp[2], qp[3]])
    barg = np.array([qp[0] - h, qp[1], qp[2], qp[3]])
    dp1dt = (H(*farg) - H(*barg)) / (2 * h)

    farg = np.array([qp[0], qp[1] + h, qp[2], qp[3]])
    barg = np.array([qp[0], qp[1] - h, qp[2], qp[3]])
    dp2dt = (H(*farg) - H(*barg)) / (2 * h)

    farg = np.array([qp[0], qp[1], qp[2] + h, qp[3]])
    barg = np.array([qp[0], qp[1], qp[2] - h, qp[3]])
    dq1dt = (H(*farg) - H(*barg)) / (2 * h)

    farg = np.array([qp[0], qp[1], qp[2], qp[3] + h])
    barg = np.array([qp[0], qp[1], qp[2], qp[3] - h])
    dq2dt = (H(*farg) - H(*barg)) / (2 * h)

    S = np.array([dq1dt, dq2dt, -dp1dt, -dp2dt])

    return S


def give_coord(qp0, frame, fps=24, calc_qp=True):
    if calc_qp == True:
        q1, q2, p1, p2 = \
            solve_ivp(fun=S_dp, t_span=[0, frame / fps], y0=qp0,
                      t_eval=np.linspace(0, frame / fps, frame, endpoint=True))[
                'y']
    else:
        q1, q2, p1, p2 = calc_qp

    x1 = np.sin(q1)
    y1 = -np.cos(q1)
    x2 = x1 + np.sin(q2)
    y2 = y1 - np.cos(q2)

    x = np.concatenate(([x1], [x2]), axis=0)
    y = np.concatenate(([y1], [y2]), axis=0)
    return x, y


def give_video_dp(initial_state, path):
    start_time = time.time()
    frame = 364
    fps = 364
    q1, q2, p1, p2 = hnn2.predict([0, frame / fps], initial_state,
                                  t_eval=np.linspace(0, frame / fps, frame, endpoint=True))

    x, y = give_coord(initial_state, frame=frame, fps=fps)
    x_hnn, y_hnn = give_coord(None, frame, fps, [q1, q2, p1, p2])
    time_traectory = time.time() - start_time
    start_time = time.time()

    path = create_film(path, x, y, x_hnn, y_hnn, frame, initial_state=initial_state)
    return path, time_traectory, time.time() - start_time


hnn = HNN.load('../../Downloads/HamiltonNeuralNetwork/HamiltonianNeuralNetwork/model_single_pendulum.npy')

hnn2 = HNND.load('../../Downloads/HamiltonNeuralNetwork/HamiltonianNeuralNetwork/model_double_pendulum.npy')
