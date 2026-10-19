#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt
import numpy as np


class EMGRadarUI(Node):

    def __init__(self):
        super().__init__('emg_radar_ui')

        # Number of EMG channels
        self.num_channels = 8

        # Latest activity magnitude for each channel
        self.emg_values = [0.0] * self.num_channels

        # Track whether data has been received
        self.received_data = False

        # Subscribe to the EMG topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/mindrove/emg',
            self.emg_callback,
            10
        )

        # Set up radar angles
        self.angles = [
            i * 2 * math.pi / self.num_channels
            for i in range(self.num_channels)
        ]

        # Close the polygon
        self.angles += self.angles[:1]

        plt.ion()

        self.fig, self.ax = plt.subplots(
        figsize=(7, 7)
        )

        self.ax.set_aspect('equal')

        self.fig.canvas.manager.set_window_title(
            'MindRove EMG Radar'
        )

        plt.show(block=False)

        self.ax.set_xticks(self.angles[:-1])
        self.ax.set_xticklabels([
            f'Channel {i + 1}'
            for i in range(self.num_channels)
        ])

        # Set up the octagonal radar chart
        self.ax.set_title(
            'MindRove EMG - Baseline-Corrected Activity',
            pad=20
        )

        # Eight equally spaced channel positions
        angles = np.linspace(
            0,
            2 * math.pi,
            self.num_channels,
            endpoint=False
        )

        # Start at the top and proceed clockwise
        angles = math.pi / 2 - angles

        # Octagon vertices
        self.angles = angles.tolist()
        self.angles += self.angles[:1]

        # Label each channel
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for i, angle in enumerate(angles):
            self.ax.text(
                1.12 * math.cos(angle),
                1.12 * math.sin(angle),
                f'Channel {i + 1}',
                ha='center',
                va='center'
            )

        # Draw the octagonal grid
        for radius in [0.25, 0.5, 0.75, 1.0]:
            grid_x = radius * np.cos(self.angles)
            grid_y = radius * np.sin(self.angles)

            self.ax.plot(
                grid_x,
                grid_y,
                linestyle='--',
                linewidth=0.8,
                alpha=0.5
            )

        # Draw the outer octagon
        self.ax.plot(
            np.cos(self.angles),
            np.sin(self.angles),
            linewidth=1.2
        )

        # Draw lines from the center to each channel
        for angle in angles:
            self.ax.plot(
                [0, math.cos(angle)],
                [0, math.sin(angle)],
                linewidth=0.8,
                alpha=0.5
            )

        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)
        self.ax.axis('off')

        # Initialize radar polygon
        self.line, = self.ax.plot(
            [0.0] * (self.num_channels + 1),
            [0.0] * (self.num_channels + 1),
            linewidth=2
        )

        self.fill = None

        self.get_logger().info(
            'EMG Radar UI started. Waiting for /mindrove/emg...'
        )

    def emg_callback(self, msg):

        data = np.asarray(msg.data, dtype=np.float64)

        # Validate the message
        if len(data) == 0 or len(data) % self.num_channels != 0:
            self.get_logger().warning(
                f'Unexpected EMG message length: {len(data)}'
            )
            return

        # Determine the number of samples per channel
        samples_per_channel = len(data) // self.num_channels

        # The publisher sends each channel's samples consecutively.
        channels = data.reshape(
            self.num_channels,
            samples_per_channel
        )

        # Remove the mean baseline independently for each channel.
        centered_channels = (
            channels
            - np.mean(channels, axis=1, keepdims=True)
        )

        # Calculate average absolute activity after centering.
        magnitudes = np.mean(
            np.abs(centered_channels),
            axis=1
        )

        self.emg_values = magnitudes.tolist()
        self.received_data = True

    def update_plot(self):

        if not self.received_data:
            return

        # Normalize activity to fit inside the octagon.
        maximum = max(self.emg_values)

        if maximum > 0:
            normalized = [
                value / maximum
                for value in self.emg_values
            ]
        else:
            normalized = [0.0] * self.num_channels

        # Close the polygon.
        values = normalized + normalized[:1]

        # Convert radial values to Cartesian coordinates.
        x = [
            radius * math.cos(angle)
            for radius, angle in zip(values, self.angles)
        ]

        y = [
            radius * math.sin(angle)
            for radius, angle in zip(values, self.angles)
        ]

        self.line.set_data(x, y)

        # Remove the previous filled area.
        if self.fill is not None:
            self.fill.remove()

        # Fill the current polygon.
        self.fill = self.ax.fill(
            x,
            y,
            alpha=0.25
        )[0]

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args=None):

    rclpy.init(args=args)

    node = EMGRadarUI()

    try:
        while rclpy.ok() and plt.fignum_exists(node.fig.number):
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update_plot()
            plt.pause(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        plt.close('all')


if __name__ == '__main__':
    main()