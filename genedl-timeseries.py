import argparse
import pandas as pd
import numpy as np
import scipy.interpolate
import os
import wave
import subprocess

def read_csv_files(file_paths):
    ratings = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df['Normalized_Rating'] = (df['Rating'] - df['Rating'].min()) / (df['Rating'].max() - df['Rating'].min())
        ratings.append(df[['Sample', 'Normalized_Rating']])
    return ratings, file_paths

def interpolate_ratings(ratings, samples):
    interpolated_ratings = []
    for rating_df in ratings:
        interp_func = scipy.interpolate.interp1d(rating_df['Sample'], rating_df['Normalized_Rating'], fill_value="extrapolate")
        interpolated_ratings.append(interp_func(samples))
    return np.array(interpolated_ratings)

def initialize_dp_table(interpolated_ratings):
    num_samples = interpolated_ratings.shape[1]
    num_takes = interpolated_ratings.shape[0]
    dp = np.full((num_samples, num_takes, 2), -np.inf, dtype=object)  # (score, last_edit_sample)
    for take in range(num_takes):
        dp[0][take] = (interpolated_ratings[take, 0], 0)
    return dp

def compute_optimal_path(dp, interpolated_ratings, min_edit_length, edit_penalty, samples):
    num_samples = len(dp)
    num_takes = len(dp[0])
    
    for sample_idx in range(1, num_samples):
        for take in range(num_takes):
            for prev_take in range(num_takes):
                score, last_edit_sample = dp[sample_idx - 1][prev_take]
                if take != prev_take and (samples[sample_idx] - last_edit_sample) < min_edit_length:
                    continue
                
                new_score = score + interpolated_ratings[take, sample_idx]
                new_last_edit_sample = last_edit_sample
                if take != prev_take:
                    new_score -= edit_penalty  # Penalty for an edit
                    new_last_edit_sample = samples[sample_idx]
                
                if new_score > dp[sample_idx][take][0]:
                    dp[sample_idx][take] = (new_score, new_last_edit_sample)
                    
    return dp

def backtrack_optimal_path(dp, samples, interpolated_ratings, min_edit_length, edit_penalty):
    num_samples = len(dp)
    num_takes = len(dp[0])
    
    max_score = -np.inf
    best_take = -1
    
    for take in range(num_takes):
        score, _ = dp[-1][take]
        if score > max_score:
            max_score = score
            best_take = take
            
    path = [best_take]
    for sample_idx in range(num_samples - 1, 0, -1):
        for prev_take in range(num_takes):
            score, last_edit_sample = dp[sample_idx - 1][prev_take]
            new_score = score + interpolated_ratings[path[-1], sample_idx]
            if path[-1] != prev_take:
                new_score -= edit_penalty  # Penalty for an edit
                if samples[sample_idx] - last_edit_sample < min_edit_length:
                    continue
            if new_score == dp[sample_idx][path[-1]][0]:
                path.append(prev_take)
                break
                
    return path[::-1], max_score

def get_wav_duration_in_samples(wav_path, sample_rate):
    result = subprocess.run(
        ['ffmpeg', '-i', wav_path, '-hide_banner', '-loglevel', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    duration = float(result.stdout)
    return int(duration * sample_rate)

def write_edl_file(edits, filenames, output_edl, sample_rate=44100, output_channels=2, title="EDL"):
    with open(output_edl, 'w') as edl_file:
        # Write the EDL header
        edl_file.write("Samplitude EDL File Format Version 1.5\n")
        edl_file.write(f'Title: "{title}"\n')
        edl_file.write(f"Sample Rate: {sample_rate}\n")
        edl_file.write(f"Output Channels: {output_channels}\n\n")

        # Write the source table entries
        edl_file.write(f"Source Table Entries: {len(filenames)}\n")
        for idx, filename in enumerate(filenames, 1):
            edl_file.write(f'   {idx} "{filename}"\n')
        edl_file.write("\n")

        # Write the track header
        edl_file.write('Track 1: "Main" Solo: 0 Mute: 0\n')
        edl_file.write("#Source Track Play-In     Play-Out    Record-In   Record-Out  Vol(dB)  MT LK FadeIn       %     CurveType                          FadeOut      %     CurveType                          Name\n")
        edl_file.write("#------ ----- ----------- ----------- ----------- ----------- -------- -- -- ------------ ----- ---------------------------------- ------------ ----- ---------------------------------- -----\n")

        # Write the edits
        for idx, (sample, take) in enumerate(edits):
            source_idx = filenames.index(take) + 1  # Get the source index
            play_in = sample
            if idx + 1 < len(edits):
                play_out = edits[idx + 1][0]
            else:
                play_out = get_wav_duration_in_samples(take)  # Extend till the end of the take if it's the last item
            
            record_in = sample
            record_out = play_out
            edl_file.write(f"      {source_idx}     1   {play_in:11d}   {play_out:11d}   {record_in:11d}   {record_out:11d}     0.00  0  0            0     0 \"*default\"                                    0     0 \"*default\"                         \"{take}\"\n")

def main(csv_files, min_edit_length, edit_penalty, output_edl):
    ratings, filenames = read_csv_files(csv_files)
    
    # Determine the range of samples
    all_samples = sorted(set().union(*(df['Sample'] for df in ratings)))
    interpolated_ratings = interpolate_ratings(ratings, all_samples)
    
    dp = initialize_dp_table(interpolated_ratings)
    dp = compute_optimal_path(dp, interpolated_ratings, min_edit_length, edit_penalty, all_samples)
    path, max_score = backtrack_optimal_path(dp, all_samples, interpolated_ratings, min_edit_length, edit_penalty)
    
    # Map indices to filenames with full path
    path_filenames = [filenames[i] for i in path]
    
    # Output necessary edits and their sample stamps
    edits = [(0, path_filenames[0])]  # Ensure the first edit point is at sample 0
    current_take = path_filenames[0]
    for i in range(1, len(path_filenames)):
        if path_filenames[i] != current_take:
            edits.append((all_samples[i], path_filenames[i]))
            current_take = path_filenames[i]
    
    # Write the EDL file with full paths
    write_edl_file(edits, filenames, output_edl)
    
    for sample, take in edits:
        print(f"Edit at sample {sample}: switch to {take}")
    
    print("Maximum Score:", max_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find optimal edit path for audio takes.')
    parser.add_argument('csv_files', type=str, nargs='+', help='List of CSV files with ratings.')
    parser.add_argument('--min_edit_length', type=float, required=True, help='Minimum edit length in samples.')
    parser.add_argument('--edit_penalty', type=float, default=0.25, help='Penalty for each edit (default: 0.25).')
    parser.add_argument('--output_edl', type=str, required=True, help='Output EDL file.')
    args = parser.parse_args()
    
    main(args.csv_files, args.min_edit_length, args.edit_penalty, args.output_edl)
