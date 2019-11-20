import time
import numpy as np
from multiprocessing import Value, Process


class OneDTargetVisualizer:
    def __init__(self):
        self.pos = Value("d", 0)
        self.target = Value("d", 0)
        self.plot_running = Value("i", 1)
        self.pp = Process(target=OneDTargetVisualizer.plot, args=(self.target, self.pos, self.plot_running))
        self.pp.start()

    def write(self, current_pos, target):
        self.pos.value = current_pos
        self.target.value = target

    def close(self):
        self.plot_running.value = 0
        time.sleep(2)
        self.pp.join()

    @staticmethod
    def plot(target, pos, plot_running):
        print("Starting visualizer")
        import matplotlib.pyplot as plt
        plt.ion()
        time.sleep(1.0)
        fig = plt.figure()
        ax1 = plt.gca()
        ax1.set_xlim(xmin=-np.pi, xmax=np.pi)
        ax1.set_ylim(ymin=-1.0, ymax=1.0)

        # target
        hl1, = ax1.plot([], [], markersize=10, marker='o', color='r')
        # position
        hl2, = ax1.plot([], [], markersize=10, marker='x', color='b')

        while plot_running.value:
            hl1.set_ydata([0])
            hl1.set_xdata([target.value])

            hl2.set_ydata([0])
            hl2.set_xdata([pos.value])

            time.sleep(0.01)
            fig.canvas.draw()
            fig.canvas.flush_events()
