import argparse
import sys
import time
import threading
import array
from pynput import keyboard
import pyaudio
import wave
import os

def play_audio(audio_file, playback_done, stop_flag):
    # Open the audio file
    wf = wave.open(audio_file, 'rb')

    # Get the sample rate, number of channels, and sample width
    sample_rate = wf.getframerate()
    channels = wf.getnchannels()
    sample_width = wf.getsampwidth()

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Define callback for playback
    def callback(in_data, frame_count, time_info, status):
        if stop_flag.is_set():
            return (None, pyaudio.paComplete)
        data = wf.readframes(frame_count)
        if len(data) == 0:
            playback_done.set()
            return (None, pyaudio.paComplete)
        return (data, pyaudio.paContinue)

    # Open stream using callback
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    stream_callback=callback)

    # Start the stream
    stream.start_stream()

    # Wait for stream to finish
    while stream.is_active():
        if stop_flag.is_set():
            break
        time.sleep(0.1)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

def record_ratings(file_path, output_file, start_measure):
    # List to store raw ratings and sample numbers
    ratings_samples = []

    playback_done = threading.Event()
    stop_flag = threading.Event()

    # Audio file
    wf = wave.open(file_path, 'rb')
    sample_rate = wf.getframerate()
    wf.close()
    playback_thread = threading.Thread(target=play_audio, args=(file_path, playback_done, stop_flag))

    def on_key_event(sample_rate, start_time):
        def inner(key):
            try:
                key_name = key.char
            except AttributeError:
                key_name = str(key)

            current_time = time.perf_counter() - start_time
            current_sample = int(current_time * sample_rate)

            if key_name == "c":
                stop_flag.set()
                playback_done.set()
                print("Stopping playback and exiting...")
                return

            if key_name and key_name.isdigit():
                rating = int(key_name)
                ratings_samples.append((current_sample, rating))

        return inner

    # Register the key press event handler
    start_time = time.perf_counter()
    listener = keyboard.Listener(on_press=on_key_event(sample_rate, start_time))
    listener.start()

    # Start playback in a separate thread
    playback_thread.start()

    try:
        playback_done.wait()
    except KeyboardInterrupt:
        print("Playback interrupted by user")
    finally:
        # Stop the listener and join the playback thread
        listener.stop()
        listener.join()  # Ensure the listener thread is joined before proceeding
        playback_thread.join()
        print("Playback finished or interrupted")

        # Process recorded ratings and sample numbers
        processed_ratings_samples = []
        last_sample = None

        for sample, rating in ratings_samples:
            if last_sample is None:
                last_sample = (sample, rating)
                continue

            # Ensure samples are not too close together
            if sample - last_sample[0] < sample_rate:
                # If they are less than 1 second apart, keep the first sample's timestamp but use the second's rating
                last_sample = (last_sample[0], rating)
            else:
                processed_ratings_samples.append(last_sample)
                last_sample = (sample, rating)

        if last_sample is not None:
            processed_ratings_samples.append(last_sample)

        # Convert processed ratings and samples to record_in and record_out
        record_ins = array.array('L', (0,) + tuple(s for s, _ in processed_ratings_samples))
        record_outs = array.array('L', (record_ins[i] for i in range(1, len(record_ins))))
        record_outs.append(record_ins[-1])
        ratings = array.array('B', (r for _, r in processed_ratings_samples))
        measure_numbers = array.array('L', (start_measure + i for i in range(len(record_ins))))

        # Save ratings to a CSV file
        with open(output_file, 'w') as f:
            f.write("Sample Rate,Measure Number,Record In,Record Out,Rating\n")
            for measure_number, record_in, record_out, rating in zip(measure_numbers, record_ins, record_outs, ratings):
                f.write(f"{sample_rate},{measure_number},{record_in},{record_out},{rating}\n")
        print(f"Ratings saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Record ratings for audio or video file.')
    parser.add_argument('input_file', type=str, nargs='?', help='Path to the input file')
    parser.add_argument('--start_measure', type=int, default=1, help='Number of the first measure')
    args = parser.parse_args()

    if not args.input_file:
        parser.print_usage()
        sys.exit(1)

    input_file = args.input_file
    output_file = os.path.splitext(input_file)[0] + '.csv'

    record_ratings(input_file, output_file, args.start_measure)

if __name__ == '__main__':
    main()
