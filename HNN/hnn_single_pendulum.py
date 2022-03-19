import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp


# Создание модели HNN
# Реализуем ее с помощью класса, так как у нас не классический учет ошибки


class HNN(tf.keras.Model):
    """
    HNN for single pendulum. It have one hidden layer. When creating hnn model,
    you must give a list of input shapes and hidden shapes.
    """

    def __init__(self, shapes, activation='sigmoid'):
        super(HNN, self).__init__()
        self.input_layer = tf.keras.layers.Input(shapes[0])
        self.hidden_layer = tf.keras.layers.Dense(shapes[1], activation=activation)
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')  # Один выходной слой, мы создаем гамильтониан

        self.W = tf.constant(np.concatenate([np.eye(shapes[0])[shapes[0] // 2:],
                                             -np.eye(shapes[0])[:shapes[0] // 2]], axis=0))
        self.history_loss = []

    def call(self, x, training=None, mask=None):
        return self.output_layer(self.hidden_layer(x))

    def forward(self, x):
        with tf.GradientTape() as tape:
            y = self.hidden_layer(x)
            y = self.output_layer(y)

        y = tape.gradient(y, x)
        return y @ self.W


def fit_hnn(model, x, y, learning_rate=1e-4, epochs=100, verbose=10):
    loss_obj = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for i in range(epochs):

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            predictions = model.forward(tf.Variable(x))
            loss = loss_obj(tf.Variable(y), predictions)

        grads = tape.gradient(loss, model.trainable_variables, unconnected_gradients='zero')
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        model.history_loss.append(loss.numpy())

        if verbose > 0:
            if i % verbose == 0:
                print(f' Step {i}: MSE: {loss.numpy() : 0.6f}')

    print(f'Last train Loss: {loss.numpy() : 0.6f}')

    return model


def integrate_hnn(model, t, y0, shapes, **kwargs):
    def s_h(t, x):
        x = tf.Variable(tf.reshape(x, (1, shapes[0])), dtype='double')
        return model.forward(x)

    return solve_ivp(s_h, t, y0, **kwargs)


args_of_pend = {'l': 2, 'm': 1, 'g': 9.8}


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


def give_data(size=9, frames=72, qstarts=(-0.4, 0., 0.4), pstarts=(1., 1., 1.), length_pend=2, drop=None):
    time_end = frames / 24
    packs = []
    params = tf.random.shuffle(np.array(np.meshgrid(qstarts, pstarts)).reshape(
        (2, len(qstarts) * len(pstarts))).T)  # Декартово произведение всех состояний

    if drop is not None:
        params = params[:drop]
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

    return np.concatenate(np.array(packs), axis=0)
