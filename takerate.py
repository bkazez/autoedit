import argparse
import subprocess
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

def play_video(video_file, playback_done, stop_flag):
    # Define the VLC command
    vlc_command = [
        'vlc', '--intf', 'dummy', '--no-video-title-show',
        '--play-and-exit', video_file
    ]

    # Start VLC process
    vlc_process = subprocess.Popen(vlc_command)

    # Wait for the video to finish
    while vlc_process.poll() is None:
        if stop_flag.is_set():
            vlc_process.terminate()
            break
        time.sleep(0.1)

    # Set playback_done flag
    playback_done.set()

def record_ratings(file_path, output_file):
    # List to store ratings and sample numbers
    ratings = array.array('L')  # Unsigned long for sample numbers
    rating_values = array.array('B')  # Unsigned char for rating values (0-9)

    playback_done = threading.Event()
    stop_flag = threading.Event()

    # Determine if the file is audio or video and get the sample rate
    if file_path.lower().endswith(('.wav', '.mp3', '.flac', '.aac')):
        # Audio file
        wf = wave.open(file_path, 'rb')
        sample_rate = wf.getframerate()
        wf.close()
        playback_thread = threading.Thread(target=play_audio, args=(file_path, playback_done, stop_flag))
    else:
        # Video file
        result_sample_rate = subprocess.run(
            ['ffmpeg', '-i', file_path, '-hide_banner', '-loglevel', 'error', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        sample_rate = int(result_sample_rate.stdout.strip().split(b'\n')[0])
        playback_thread = threading.Thread(target=play_video, args=(file_path, playback_done, stop_flag))

    def on_key_event(sample_rate, start_time):
        def inner(key):
            try:
                key_name = key.char
            except AttributeError:
                key_name = str(key)

            # Check for character "c"
            if key_name == "c":
                stop_flag.set()
                playback_done.set()
                print("Stopping playback and exiting...")

            if key_name and key_name.isdigit():
                rating = int(key_name)
                current_time = time.time() - start_time
                current_sample = int(current_time * sample_rate)
                ratings.append(current_sample)
                rating_values.append(rating)
                print(f"\bRecorded rating: {rating} at sample {current_sample}")

        return inner

    # Register the key press event handler
    start_time = time.time()
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

        # Save ratings to a CSV file
        with open(output_file, 'w') as f:
            f.write("Sample,Rating,Sample Rate\n")  # Write the header
            for sample, rating in zip(ratings, rating_values):
                f.write(f"{sample},{rating},{sample_rate}\n")
        print(f"Ratings saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Record ratings for audio or video file.')
    parser.add_argument('input_file', type=str, nargs='?', help='Path to the input file')
    args = parser.parse_args()

    if not args.input_file:
        parser.print_usage()
        sys.exit(1)

    input_file = args.input_file
    output_file = input_file.rsplit('.', 1)[0] + '.csv'

    record_ratings(input_file, output_file)

if __name__ == '__main__':
    if not os.geteuid() == 0:
        print("This script must be run as root. Use sudo.")
        sys.exit(1)
    main()