# Author: Akifumi Wachi
# Copyright 2021- IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT

import sys
import numpy as np
from matplotlib import pyplot as plt
from gym_minigrid.window import Window


class MyWindow(Window):
    """
    Window to draw a gridworld instance using Matplotlib
    """
    def __init__(self, title) -> None:

        self.fig = None

        self.imshow_obj = None

        N_X = 2
        N_Y = 3

        # Create the figure and axes
        self.fig, self.ax = plt.subplots(N_X, N_Y)

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)
        self.ax[0, 0].set_title("Agent behavior", fontsize=10)
        self.ax[0, 1].set_title("True safety", fontsize=10)
        self.ax[0, 2].set_title("True reward", fontsize=10)
        self.ax[1, 0].set_title("Optimistic safe region", fontsize=10)
        self.ax[1, 1].set_title("Pessimistic safe region", fontsize=10)
        self.ax[1, 2].set_title("Upper bound of reward", fontsize=10)

        # Turn off x/y axis numbering/ticks
        for i in range(N_X):
            for j in range(N_Y):
                self.ax[i, j].xaxis.set_ticks_position('none')
                self.ax[i, j].yaxis.set_ticks_position('none')
                _ = self.ax[i, j].set_xticklabels([])
                _ = self.ax[i, j].set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img, env, x) -> None:
        """
        Show an image or update the image being shown
        """
        if self.imshow_obj is None:
            self.imshow_obj = self.ax[0, 0].imshow(
                img, interpolation='bilinear'
                )
        
        # Agent's behavior
        self.imshow_obj.set_data(img)

        # True safety
        self.ax[0, 1].imshow(
            env.true_safety_map.T, interpolation='none', cmap='gray'
            )

        # True reward
        self.ax[0, 2].imshow(env.reward[:, :].T)

        if x.sim_type == 'oracle':
            # Upper bound of reward
            self.ax[1, 2].imshow(env.reward[:, :].T)

            # Optimistic safe region = True safe region
            self.ax[1, 0].imshow(
                env.true_safety_map.T, interpolation='none', cmap='gray'
                )

            # Pessimistic safe region = True safe region
            self.ax[1, 1].imshow(
                env.true_safety_map.T, interpolation='none', cmap='gray'
                )

        else:
            # Upper bound of reward
            self.ax[1, 2].imshow(x.bounds_reward[:, :, 0].T)

            # Optimistic safe region
            self.ax[1, 0].imshow(
                x.safety_reachability_map[:, :, 0].T, 
                interpolation='none', cmap='gray'
                )

            # Pessimistic safe region
            self.ax[1, 1].imshow(
                x.safety_reachability_map[:, :, 1].T,
                interpolation='none', cmap='gray'
                )

        self.fig.canvas.draw()
        plt.pause(0.0001)
