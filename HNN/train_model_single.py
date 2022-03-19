import time

from hnn_single_pendulum import *

start = time.time()
if __name__ == '__main__':
    data = give_data(size=50, frames=72, qstarts=np.linspace(-np.pi / 2, np.pi / 2, 5, endpoint=True),
                     pstarts=np.linspace(-50, 50, 10, endpoint=True))
    x = data[:, :2]
    y = data[:, 2:]

    hnn = HNN((2, 200, 1))
    hnn = fit_hnn(hnn, x, y, 0.0005, 2000, 200)
    print(hnn.history_loss)

print(time.time() - start)
