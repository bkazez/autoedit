import argparse
import pandas as pd
import numpy as np
import os

def read_csv_files(file_paths):
    ratings = []
    sample_rates = []
    wav_files = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        sample_rates.append(df['Sample Rate'].iloc[0])
        df['Normalized_Rating'] = (df['Rating'] - df['Rating'].min()) / (df['Rating'].max() - df['Rating'].min())
        ratings.append(df[['Sample', 'Normalized_Rating']])
        wav_file = file_path.rsplit('.', 1)[0] + '.wav'
        wav_files.append(wav_file)
    return ratings, sample_rates, wav_files

def initialize_dp_table(ratings):
    num_samples = len(ratings[0])
    num_takes = len(ratings)
    dp = np.full((num_samples, num_takes, 2), -np.inf, dtype=object)  # (score, last_edit_sample)
    for take in range(num_takes):
        dp[0][take] = (ratings[take]['Normalized_Rating'].iloc[0], 0)
    return dp

def compute_optimal_path(dp, ratings, min_edit_length, edit_penalty):
    num_samples = len(dp)
    num_takes = len(dp[0])
    
    for sample_idx in range(1, num_samples):
        for take in range(num_takes):
            for prev_take in range(num_takes):
                score, last_edit_sample = dp[sample_idx - 1][prev_take]
                if take != prev_take and (ratings[take]['Sample'].iloc[sample_idx] - last_edit_sample) < min_edit_length:
                    continue
                
                new_score = score + ratings[take]['Normalized_Rating'].iloc[sample_idx]
                new_last_edit_sample = last_edit_sample
                if take != prev_take:
                    new_score -= edit_penalty  # Penalty for an edit
                    new_last_edit_sample = ratings[take]['Sample'].iloc[sample_idx]
                
                if new_score > dp[sample_idx][take][0]:
                    dp[sample_idx][take] = (new_score, new_last_edit_sample)
                    
    return dp

def backtrack_optimal_path(dp, ratings, min_edit_length, edit_penalty):
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
    edits = []
    for sample_idx in range(num_samples - 1, 0, -1):
        for prev_take in range(num_takes):
            score, last_edit_sample = dp[sample_idx - 1][prev_take]
            new_score = score + ratings[path[-1]]['Normalized_Rating'].iloc[sample_idx]
            if path[-1] != prev_take:
                new_score -= edit_penalty  # Penalty for an edit
                if ratings[path[-1]]['Sample'].iloc[sample_idx] - last_edit_sample < min_edit_length:
                    continue
            if new_score == dp[sample_idx][path[-1]][0]:
                path.append(prev_take)
                if prev_take != path[-2]:  # Save edit point if it's a switch
                    edits.append((sample_idx, prev_take, ratings[prev_take]['Sample'].iloc[sample_idx], ratings[prev_take]['Sample'].iloc[sample_idx + 1] if sample_idx + 1 < num_samples else ratings[prev_take]['Sample'].iloc[-1]))
                break
                
    edits = edits[::-1]
    return path[::-1], max_score, edits

def print_dp_table(dp, ratings):
    num_samples = len(dp)
    num_takes = len(dp[0])
    for sample_idx in range(num_samples):
        print(f"Sample {sample_idx}:")
        for take in range(num_takes):
            score, last_edit_sample = dp[sample_idx][take]
            if score != -np.inf:
                print(f"  Take {take}: score={score:.2f}, last_edit_sample={last_edit_sample}")

def write_edl_file(edits, filenames, sample_rates, output_edl, output_channels=2, title="EDL"):
    with open(output_edl, 'w') as edl_file:
        # Write the EDL header
        edl_file.write("Samplitude EDL File Format Version 1.5\n")
        edl_file.write(f'Title: "{title}"\n')
        edl_file.write(f"Sample Rate: {sample_rates[0]}\n")
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
        current_play_in = 0
        for idx, (measure_idx, take_idx, record_in, record_out) in enumerate(edits):
            take = filenames[take_idx]
            source_idx = filenames.index(take) + 1  # Get the source index
            duration = record_out - record_in
            play_out = current_play_in + duration

            edl_file.write(f"      {source_idx}     1   {current_play_in:11d}   {play_out:11d}   {record_in:11d}   {record_out:11d}     0.00  0  0            0     0 \"*default\"                                    0     0 \"*default\"                         \"{take}\"\n")
            current_play_in = play_out

def main(csv_files, min_edit_length, edit_penalty, output_edl):
    ratings, sample_rates, filenames = read_csv_files(csv_files)

    dp = initialize_dp_table(ratings)
    dp = compute_optimal_path(dp, ratings, min_edit_length, edit_penalty)

    # Print all edit paths considered
    print_dp_table(dp, ratings)

    path, max_score, edits = backtrack_optimal_path(dp, ratings, min_edit_length, edit_penalty)
    
    # Add the first edit to start at sample 0 with the best initial take
    initial_take = path[0]
    edits.insert(0, (0, initial_take, 0, ratings[initial_take]['Sample'].iloc[1]))
    
    # Adjust the last edit to extend the best final take to the end
    final_take = path[-1]
    last_edit_idx = len(edits) - 1
    edits[last_edit_idx] = (edits[last_edit_idx][0], edits[last_edit_idx][1], edits[last_edit_idx][2], ratings[final_take]['Sample'].iloc[-1])

    # Write the EDL file with full paths
    write_edl_file(edits, filenames, sample_rates, output_edl)
    
    for measure_idx, take_idx, record_in, record_out in edits:
        print(f"Edit at measure {measure_idx}: switch to {filenames[take_idx]}, record in {record_in}, record out {record_out}")
    
    print("Maximum Score:", max_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find optimal edit path for audio takes.')
    parser.add_argument('csv_files', type=str, nargs='+', help='List of CSV files with ratings.')
    parser.add_argument('--min_edit_length', type=float, required=True, help='Minimum edit length in samples.')
    parser.add_argument('--edit_penalty', type=float, default=0.25, help='Penalty for each edit (default: 0.25).')
    parser.add_argument('--output_edl', type=str, required=True, help='Output EDL file.')
    args = parser.parse_args()
    
    main(args.csv_files, args.min_edit_length, args.edit_penalty, args.output_edl)
