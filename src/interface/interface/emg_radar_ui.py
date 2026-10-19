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


        # EMG CONFIGURATION
        self.num_channels = 8

        # Latest activity magnitude for each channel
        self.emg_values = [0.0] * self.num_channels
        self.received_data = False

        # Subscribe to EMG data
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/mindrove/emg',
            self.emg_callback,
            10
        )


        # MACHINE LEARNING PLACEHOLDERS
        self.class_names = [
            'Relaxed',
            'Fist',
            'Wrist\nFlexion',
            'Wrist\nExtension'
        ]

        # DEMONSTRATION VALUES ONLY.
        # These are not predictions from actual ML models.

        self.lda_confidence = [
            0.80,  # Relaxed
            0.15,  # Fist
            0.10,  # Wrist Flexion
            0.05   # Wrist Extension
        ]

        self.svm_confidence = [
            0.75,  # Relaxed
            0.20,  # Fist
            0.15,  # Wrist Flexion
            0.10   # Wrist Extension
        ]


        # APPLICATION WINDOW
        plt.ion()

        self.fig = plt.figure(figsize=(14, 8))

        self.fig.canvas.manager.set_window_title(
            'MindRove EMG Interface'
        )

        grid = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[1.2, 1.0],
            wspace=0.30
        )

        # Left: EMG radar
        self.ax = self.fig.add_subplot(grid[0, 0])

        # Right: Classification confidence
        self.conf_ax = self.fig.add_subplot(grid[0, 1])

        self.fig.suptitle(
            'MindRove EMG Interface',
            fontsize=16,
            fontweight='bold'
        )


        # LEFT PANEL: OCTAGONAL EMG RADAR
        self.ax.set_title(
            'EMG Activity',
            fontsize=13,
            pad=20
        )

        self.ax.set_aspect('equal')

        # Eight equally spaced channel positions
        angles = np.linspace(
            0,
            2 * math.pi,
            self.num_channels,
            endpoint=False
        )

        # Start at the top and proceed clockwise
        angles = math.pi / 2 - angles

        self.angles = angles.tolist()

        # Close the polygon
        self.angles += self.angles[:1]

        # Channel labels
        for i, angle in enumerate(angles):
            self.ax.text(
                1.13 * math.cos(angle),
                1.13 * math.sin(angle),
                f'Channel {i + 1}',
                ha='center',
                va='center',
                fontsize=9
            )

        # Octagonal grid
        for radius in [0.25, 0.50, 0.75, 1.00]:

            grid_x = radius * np.cos(self.angles)
            grid_y = radius * np.sin(self.angles)

            self.ax.plot(
                grid_x,
                grid_y,
                linestyle='--',
                linewidth=0.8,
                alpha=0.5
            )

        # Outer octagon
        self.ax.plot(
            np.cos(self.angles),
            np.sin(self.angles),
            linewidth=1.2
        )

        # Lines from the center to each channel
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
        self.radar_line, = self.ax.plot(
            [0.0] * (self.num_channels + 1),
            [0.0] * (self.num_channels + 1),
            linewidth=2
        )

        self.radar_fill = None

        # Classification result beneath the radar
        self.result_text = self.ax.text(
            0.5,
            -0.05,
            'Waiting for classification...',
            transform=self.ax.transAxes,
            ha='center',
            va='top',
            fontsize=11,
            fontweight='bold',
            linespacing=1.6
        )


        # RIGHT PANEL: LDA AND SVM CONFIDENCE
        self.conf_ax.set_title(
            'Classification Confidence',
            fontsize=13,
            pad=25
        )

        x_positions = np.arange(len(self.class_names))

        bar_width = 0.32

        # Place LDA and SVM beside each other
        self.lda_positions = x_positions - bar_width / 2
        self.svm_positions = x_positions + bar_width / 2

        # LDA pillars
        self.lda_bars = self.conf_ax.bar(
            self.lda_positions,
            self.lda_confidence,
            width=bar_width,
            label='LDA'
        )

        # SVM pillars
        self.svm_bars = self.conf_ax.bar(
            self.svm_positions,
            self.svm_confidence,
            width=bar_width,
            label='SVM'
        )

        # Fixed percentage scale
        self.conf_ax.set_ylim(0, 1.0)

        self.conf_ax.set_yticks(
            np.linspace(0, 1, 6)
        )

        self.conf_ax.set_yticklabels([
            '0%',
            '20%',
            '40%',
            '60%',
            '80%',
            '100%'
        ])

        # Classification labels centered under each pair
        self.conf_ax.set_xticks(x_positions)

        self.conf_ax.set_xticklabels(
            self.class_names,
            fontsize=9
        )

        self.conf_ax.set_ylabel('Confidence')

        # Horizontal grid
        self.conf_ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.4
        )

        self.conf_ax.set_axisbelow(True)

        # Legend identifying each model
        self.conf_ax.legend(
            loc='upper right'
        )

        # Percentage labels above LDA pillars
        self.lda_labels = []

        for position, value in zip(
            self.lda_positions,
            self.lda_confidence
        ):
            label = self.conf_ax.text(
                position,
                value + 0.025,
                f'{value:.0%}',
                ha='center',
                va='bottom',
                fontsize=8
            )

            self.lda_labels.append(label)

        # Percentage labels above SVM pillars
        self.svm_labels = []

        for position, value in zip(
            self.svm_positions,
            self.svm_confidence
        ):
            label = self.conf_ax.text(
                position,
                value + 0.025,
                f'{value:.0%}',
                ha='center',
                va='bottom',
                fontsize=8
            )

            self.svm_labels.append(label)

        # Explicit placeholder warning
        self.conf_ax.text(
            0.5,
            1.08,
            'DEMO VALUES - MODELS NOT CONNECTED',
            transform=self.conf_ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=9
        )

        self.conf_ax.set_xlim(
            -0.6,
            len(self.class_names) - 0.4
        )

        # Display application
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show(block=False)

        self.get_logger().info(
            'EMG interface started. Waiting for /mindrove/emg...'
        )


    # EMG DATA PROCESSING
    def emg_callback(self, msg):

        data = np.asarray(
            msg.data,
            dtype=np.float64
        )

        # Validate message
        if (
            len(data) == 0
            or len(data) % self.num_channels != 0
        ):
            self.get_logger().warning(
                f'Unexpected EMG message length: {len(data)}'
            )
            return

        # Determine samples per channel
        samples_per_channel = (
            len(data) // self.num_channels
        )

        # Reconstruct individual channels
        channels = data.reshape(
            self.num_channels,
            samples_per_channel
        )

        # Remove the mean baseline independently
        # for each channel
        centered_channels = (
            channels
            - np.mean(
                channels,
                axis=1,
                keepdims=True
            )
        )

        # Calculate average absolute activity
        magnitudes = np.mean(
            np.abs(centered_channels),
            axis=1
        )

        self.emg_values = magnitudes.tolist()
        self.received_data = True


    # UPDATE RADAR CHART
    def update_radar(self):

        if not self.received_data:
            return

        # Normalize activity to fit inside the octagon
        maximum = max(self.emg_values)

        if maximum > 0:
            normalized = [
                value / maximum
                for value in self.emg_values
            ]
        else:
            normalized = [
                0.0
            ] * self.num_channels

        # Close polygon
        values = normalized + normalized[:1]

        # Convert radial values to Cartesian coordinates
        x = [
            radius * math.cos(angle)
            for radius, angle in zip(
                values,
                self.angles
            )
        ]

        y = [
            radius * math.sin(angle)
            for radius, angle in zip(
                values,
                self.angles
            )
        ]

        self.radar_line.set_data(x, y)

        # Remove previous filled area
        if self.radar_fill is not None:
            self.radar_fill.remove()

        # Fill current polygon
        self.radar_fill = self.ax.fill(
            x,
            y,
            alpha=0.25
        )[0]


    # UPDATE CONFIDENCE PILLARS
    def update_confidence(self):

        # Update LDA pillars
        for bar, label, value in zip(
            self.lda_bars,
            self.lda_labels,
            self.lda_confidence
        ):
            bar.set_height(value)

            label.set_position((
                bar.get_x() + bar.get_width() / 2,
                value + 0.025
            ))

            label.set_text(
                f'{value:.0%}'
            )

        # Update SVM pillars
        for bar, label, value in zip(
            self.svm_bars,
            self.svm_labels,
            self.svm_confidence
        ):
            bar.set_height(value)

            label.set_position((
                bar.get_x() + bar.get_width() / 2,
                value + 0.025
            ))

            label.set_text(
                f'{value:.0%}'
            )


    def update_classification_text(self):

        # Find the highest-confidence classification for each model
        lda_index = int(np.argmax(self.lda_confidence))
        svm_index = int(np.argmax(self.svm_confidence))

        lda_class = self.class_names[lda_index].replace('\n', ' ')
        svm_class = self.class_names[svm_index].replace('\n', ' ')

        lda_value = self.lda_confidence[lda_index]
        svm_value = self.svm_confidence[svm_index]

        # Determine whether both models agree
        if lda_index == svm_index:
            overall = f'Both models agree: {lda_class}'
        else:
            overall = 'Models disagree'

        text = (
            f'LDA: {lda_class} ({lda_value:.0%})\n'
            f'SVM: {svm_class} ({svm_value:.0%})\n'
            f'Overall: {overall}'
        )

        self.result_text.set_text(text)



    # UPDATE COMPLETE INTERFACE
    def update_plot(self):

        self.update_radar()
        self.update_confidence()
        self.update_classification_text()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()



def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = EMGRadarUI()

        while (
            rclpy.ok()
            and plt.fignum_exists(node.fig.number)
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.01
            )

            node.update_plot()

            plt.pause(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        plt.close('all')


if __name__ == '__main__':
    main()