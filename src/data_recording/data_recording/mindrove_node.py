import csv
from datetime import datetime
import select
import sys
import termios
import time
import tty
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

from mindrove.board_shim import (
    BoardShim,
    MindRoveInputParams,
    BoardIds
)


class MindRoveNode(Node):

    def __init__(self):
        super().__init__('mindrove_node')

        if not sys.stdin.isatty():
            raise RuntimeError(
                'Run mindrove_node from a terminal to use keyboard recording.'
            )

        self.terminal_settings = termios.tcgetattr(sys.stdin.fileno())
        self.recording_duration = 20.0
        self.recording_deadline = None
        self.recording_filename = None
        self.recorded_samples = []
        self.data_directory = (
            Path.cwd() / 'src' / 'data_classification' / 'data'
        )
        if not self.data_directory.parent.is_dir():
            raise RuntimeError(
                'Run mindrove_node from the repository root so the '
                'data_classification package can be found.'
            )
        self.data_directory.mkdir(parents=True, exist_ok=True)


        # MindRove setup

        self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        self.params = MindRoveInputParams()

        self.board = BoardShim(
            self.board_id,
            self.params
        )

        self.get_logger().info('Connecting to MindRove...')

        self.board.prepare_session()
        self.board.start_stream()

        self.get_logger().info('MindRove connected!')
        self.get_logger().info('MindRove streaming started.')

        try:
            tty.setcbreak(sys.stdin.fileno())
        except OSError:
            self.board.stop_stream()
            self.board.release_session()
            raise

        # Get channel information
        self.eeg_channels = BoardShim.get_eeg_channels(
            self.board_id
        )

        self.emg_channels = BoardShim.get_emg_channels(
            self.board_id
        )

        self.sampling_rate = BoardShim.get_sampling_rate(
            self.board_id
        )

        self.get_logger().info(
            f'EMG channels: {self.emg_channels}'
        )

        self.get_logger().info(
            f'Sampling rate: {self.sampling_rate} Hz'
        )


        # ROS publisher
        self.emg_publisher = self.create_publisher(
            Float64MultiArray,
            '/mindrove/emg',
            10
        )

        # Timer
        # Runs at 500 Hz
        self.timer = self.create_timer(
            0.002,
            self.publish_data
        )

        self.get_logger().info(
            'Press r to record EMG data for 20 seconds.'
        )

    def publish_data(self):
        self.check_keyboard()

        count = self.board.get_board_data_count()

        if count == 0:
            self.finish_recording_if_due()
            return

        # Consume each sample once so recordings do not contain duplicates.
        data = self.board.get_board_data()

        # EMG
        if self.emg_channels:

            emg_msg = Float64MultiArray()

            for channel in self.emg_channels:
                emg_msg.data.extend(
                    data[channel, :].tolist()
                )

            self.emg_publisher.publish(emg_msg)

            if self.recording_deadline is not None:
                if time.monotonic() < self.recording_deadline:
                    self.recorded_samples.extend(
                        data[self.emg_channels, :].T.tolist()
                    )
                else:
                    self.finish_recording()

        self.finish_recording_if_due()

    def check_keyboard(self):
        if not select.select([sys.stdin], [], [], 0)[0]:
            return

        key = sys.stdin.read(1)
        if key.lower() != 'r':
            return

        if self.recording_deadline is not None:
            self.get_logger().warning(
                'A recording is already in progress.'
            )
            return

        self.recording_filename = (
            f'emg_recording_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.csv'
        )
        self.recorded_samples = []
        self.recording_deadline = time.monotonic() + self.recording_duration
        self.get_logger().info(
            f'Recording started; saving to {self.recording_filename}.'
        )

    def finish_recording_if_due(self):
        if (
            self.recording_deadline is not None
            and time.monotonic() >= self.recording_deadline
        ):
            self.finish_recording()

    def finish_recording(self):
        with open(
            self.data_directory / self.recording_filename,
            'w',
            newline='',
            encoding='utf-8'
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                f'emg_channel_{channel}' for channel in self.emg_channels
            ])
            writer.writerows(self.recorded_samples)

        self.get_logger().info(
            f'Recording complete: '
            f'{self.data_directory / self.recording_filename} '
            f'({len(self.recorded_samples)} samples).'
        )
        self.recording_deadline = None
        self.recording_filename = None
        self.recorded_samples = []

    def destroy_node(self):

        self.get_logger().info(
            'Stopping MindRove...'
        )

        try:
            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSADRAIN,
                self.terminal_settings
            )
        except OSError as error:
            self.get_logger().error(
                f'Unable to restore terminal settings: {error}'
            )

        if self.recording_deadline is not None:
            self.get_logger().warning(
                'Recording stopped before 20 seconds; partial data was not saved.'
            )

        try:
            self.board.stop_stream()
        except Exception:
            pass

        self.board.release_session()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = MindRoveNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()