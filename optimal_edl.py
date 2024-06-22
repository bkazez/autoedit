import pandas as pd
import argparse
import os
import numpy as np

def read_csv_files(filepaths):
    dfs = [pd.read_csv(filepath) for filepath in filepaths]
    return dfs

def calculate_optimal_path(dfs, edit_penalty):
    all_measures = sorted(set().union(*(df['Measure Number'].unique() for df in dfs)))
    num_measures = len(all_measures)
    num_takes = len(dfs)

    dp = np.full((num_measures, num_takes), -np.inf)
    path = np.full((num_measures, num_takes), -1, dtype=int)

    first_measure = all_measures[0]
    for take in range(num_takes):
        if first_measure in dfs[take]['Measure Number'].values:
            rating = dfs[take].loc[dfs[take]['Measure Number'] == first_measure, 'Rating'].values[0]
            dp[0, take] = rating

    for i in range(1, num_measures):
        current_measure = all_measures[i]
        for curr_take in range(num_takes):
            if current_measure in dfs[curr_take]['Measure Number'].values:
                current_rating = dfs[curr_take].loc[dfs[curr_take]['Measure Number'] == current_measure, 'Rating'].values[0]
                for prev_take in range(num_takes):
                    if dp[i-1, prev_take] != -np.inf:
                        switch_penalty = edit_penalty if curr_take != prev_take else 0
                        score = dp[i-1, prev_take] + current_rating - switch_penalty
                        if score > dp[i, curr_take]:
                            dp[i, curr_take] = score
                            path[i, curr_take] = prev_take

    end_measure = all_measures[-1]
    best_score = -np.inf
    best_end_take = -1
    for take in range(num_takes):
        if dp[-1, take] > best_score:
            best_score = dp[-1, take]
            best_end_take = take

    optimal_path = []
    current_take = best_end_take
    for i in range(num_measures-1, -1, -1):
        optimal_path.append((all_measures[i], current_take))
        current_take = path[i, current_take]
    optimal_path.reverse()

    return optimal_path, best_score

def generate_samplitude_edl(optimal_path, filepaths, dfs):
    edl_header = """Samplitude EDL File Format Version 1.5
Title: "Generated EDL"
Sample Rate: 48000
Output Channels: 2

Source Table Entries: {num_sources}
"""

    source_entries = ""
    source_ids = {}
    for i, filepath in enumerate(filepaths):
        source_ids[filepath] = i + 1
        wav_filename = filepath.replace('.csv', '.wav')
        source_entries += f"   {i+1} \"{wav_filename}\"\n"

    track_header = """Track 1: "Main" Solo: 0 Mute: 0
#Source      Track       Play-In     Play-Out    Record-In   Record-Out  Vol(dB)  MT LK FadeIn  %     CurveType    FadeOut  %     CurveType    Name
#----------- ----------- ----------- ----------- ----------- ----------- -------- -- -- ------- ----- ----------- -------- ----- ------------ -----------
"""

    track_entries = ""
    current_sample = 0
    i = 0

    while i < len(optimal_path):
        measure, take = optimal_path[i]
        df = dfs[take]
        record_in = df.loc[df['Measure Number'] == measure, 'Record In'].values[0]

        j = i
        while j < len(optimal_path) and optimal_path[j][1] == take:
            next_measure = optimal_path[j][0]
            record_out = df.loc[df['Measure Number'] == next_measure, 'Record Out'].values[0]
            j += 1

        duration = record_out - record_in
        play_in = current_sample
        play_out = play_in + duration

        wav_filename = os.path.basename(filepaths[take]).replace('.csv', '.wav')
        source_id = source_ids[filepaths[take]]
        track_entries += "{:>11d} {:>11d} {:>11d} {:>11d} {:>11d} {:>11d} {:>8} {:>2} {:>2} {:>7} {:>5}  {:<12} {:>6} {:>5}  {:<11}  {}\n".format(
            source_id, 1, play_in, play_out, record_in, record_out, "0.00", 0, 0, 0, 0, "\"*default\"", 0, 0, "\"*default\"", f"\"{wav_filename}\""
        )

        current_sample = play_out
        i = j

    edl_content = edl_header.format(num_sources=len(filepaths)) + source_entries + "\n" + track_header + track_entries
    return edl_content

def main(filepaths, edit_penalty):
    dfs = read_csv_files(filepaths)
    optimal_path, best_score = calculate_optimal_path(dfs, edit_penalty)

    take_names = [os.path.basename(filepath) for filepath in filepaths]
    
    path_takes = [take for measure, take in optimal_path]
    print(f"Optimal Path: {path_takes}            {best_score:.4f}")
    
    for measure, take in optimal_path:
        print(f'Measure: {measure}, Take: {take_names[take]}')

    edl_content = generate_samplitude_edl(optimal_path, filepaths, dfs)
    with open("output.edl", "w") as edl_file:
        edl_file.write(edl_content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the optimal edit decision list from CSV files.')
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='CSV files containing the ratings')
    parser.add_argument('--edit_penalty', type=float, default=0.1, help='Penalty for changing takes')

    args = parser.parse_args()
    main(args.files, args.edit_penalty)
