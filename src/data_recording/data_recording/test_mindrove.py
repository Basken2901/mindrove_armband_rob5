from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import time


def main():
    # Enable useful SDK messages while we are testing
    BoardShim.enable_dev_board_logger()

    # Create connection parameters
    params = MindRoveInputParams()

    # Select the MindRove WiFi board
    board_id = BoardIds.MINDROVE_WIFI_BOARD

    # Create the board object
    board = BoardShim(board_id, params)

    print("Connecting to MindRove...")

    try:
        # Connect to the armband
        board.prepare_session()
        print("Connected!")

        # Start receiving data
        board.start_stream()
        print("Streaming started!")

        # Get information about the board
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        emg_channels = BoardShim.get_emg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        print(f"EEG channels: {eeg_channels}")
        print(f"EMG channels: {emg_channels}")
        print(f"Sampling rate: {sampling_rate} Hz")

        print("\nWaiting for data...\n")

        while True:
            # Check how much data is available
            count = board.get_board_data_count()

            if count > 0:
                # Get the newest samples
                data = board.get_current_board_data(
                    min(count, sampling_rate)
                )

                print(f"Received {data.shape[1]} samples")

                if eeg_channels:
                    print("EEG:", data[eeg_channels, -1])

                if emg_channels:
                    print("EMG:", data[emg_channels, -1])

                print()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        try:
            board.stop_stream()
        except Exception:
            pass

        board.release_session()
        print("MindRove disconnected.")


if __name__ == "__main__":
    main()