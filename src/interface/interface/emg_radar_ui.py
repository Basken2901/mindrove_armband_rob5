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

        # Number of EMG channels in the incoming message
        self.num_channels = 8

        # Latest magnitude for each channel
        self.emg_values = [0.0] * self.num_channels

        # Track whether we have received any data
        self.received_data = False

        # Subscribe to the EMG topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/mindrove/emg',
            self.emg_callback,
            10
        )

        # Set up the radar chart
        self.angles = [
            i * 2 * math.pi / self.num_channels
            for i in range(self.num_channels)
        ]

        # Close the polygon
        self.angles += self.angles[:1]

        plt.ion()

        self.fig, self.ax = plt.subplots(
            figsize=(7, 7),
            subplot_kw={'polar': True}
        )

        # Give the window a title
        self.fig.canvas.manager.set_window_title(
            'MindRove EMG Radar'
        )

        # Show the window without blocking normal desktop interaction
        plt.show(block=False)

        self.ax.set_xticks(self.angles[:-1])
        self.ax.set_xticklabels([
            f'Channel {i + 1}'
            for i in range(self.num_channels)
        ])

        self.ax.set_title(
            'MindRove EMG - Absolute Magnitude',
            pad=25
        )

        self.ax.set_ylim(0, 25000)
        self.ax.grid(True)

        # Initialize the radar polygon
        self.line, = self.ax.plot(
            self.angles,
            [0.0] * (self.num_channels + 1),
            linewidth=2
        )

        self.fill = None

        self.get_logger().info(
            'EMG Radar UI started. Waiting for /mindrove/emg...'
        )

    def emg_callback(self, msg):

        data = list(msg.data)

        # Expect 8 channels with an equal number of samples per channel.
        if len(data) == 0 or len(data) % self.num_channels != 0:
            self.get_logger().warning(
                f'Unexpected EMG message length: {len(data)}'
            )
            return

        samples_per_channel = len(data) // self.num_channels

        magnitudes = []

        for channel in range(self.num_channels):

            start = channel * samples_per_channel
            end = start + samples_per_channel

            channel_samples = data[start:end]

            # Take the absolute value of every sample,
            # then calculate the average magnitude.
            magnitude = sum(
                abs(float(value))
                for value in channel_samples
            ) / samples_per_channel

            magnitudes.append(magnitude)

        self.emg_values = magnitudes
        self.received_data = True

    def update_plot(self):

        if not self.received_data:
            return

        # Repeat the first value to close the radar polygon.
        values = self.emg_values + self.emg_values[:1]

        # Update the radar line.
        self.line.set_data(self.angles, values)

        # Remove the previous filled area.
        if self.fill is not None:
            self.fill.remove()

        # Fill the current polygon.
        self.fill = self.ax.fill(
            self.angles,
            values,
            alpha=0.25
        )[0]

        # Adjust the radial scale if necessary.
        maximum = max(self.emg_values)

        if maximum > 0:
            self.ax.set_ylim(0, max(25000, maximum * 1.1))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args=None):

    rclpy.init(args=args)

    node = EMGRadarUI()

    try:
        # Process ROS messages while keeping the GUI responsive.
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