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

        # -------------------------
        # MindRove setup
        # -------------------------

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
            f'EEG channels: {self.eeg_channels}'
        )

        self.get_logger().info(
            f'EMG channels: {self.emg_channels}'
        )

        self.get_logger().info(
            f'Sampling rate: {self.sampling_rate} Hz'
        )

        # -------------------------
        # ROS publishers
        # -------------------------

        self.eeg_publisher = self.create_publisher(
            Float64MultiArray,
            '/mindrove/eeg',
            10
        )

        self.emg_publisher = self.create_publisher(
            Float64MultiArray,
            '/mindrove/emg',
            10
        )

        # -------------------------
        # Timer
        # -------------------------

        # Run at 100 Hz
        self.timer = self.create_timer(
            0.01,
            self.publish_data
        )

    def publish_data(self):

        count = self.board.get_board_data_count()

        if count == 0:
            return

        # Get the newest samples
        data = self.board.get_current_board_data(
            min(count, 10)
        )

        # -------------------------
        # EEG
        # -------------------------

        if self.eeg_channels:

            eeg_msg = Float64MultiArray()

            for channel in self.eeg_channels:
                eeg_msg.data.extend(
                    data[channel, :].tolist()
                )

            self.eeg_publisher.publish(eeg_msg)

        # -------------------------
        # EMG
        # -------------------------

        if self.emg_channels:

            emg_msg = Float64MultiArray()

            for channel in self.emg_channels:
                emg_msg.data.extend(
                    data[channel, :].tolist()
                )

            self.emg_publisher.publish(emg_msg)

    def destroy_node(self):

        self.get_logger().info(
            'Stopping MindRove...'
        )

        try:
            self.board.stop_stream()
        except Exception:
            pass

        self.board.release_session()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = MindRoveNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()