"""
Slice filtered .csv lexique into smaller files containing 500 lemmes apiece.
"""
import os

import pandas as pd


# === CONFIGURATION ===
USER_PATH = os.path.expanduser('~')
INPUT_CSV = f'{USER_PATH}/Documents/flashcard_project_new/Lexique383 - Filtered.csv'
OUTPUT_DIR = f'{USER_PATH}/Documents/flashcard_project_new/lexique_exported_files'
CHUNK_SIZE = 500
SPOKEN_COUNT = 400
WRITTEN_COUNT = 100

# === STEP 1: LOAD CSV ===
df_all = pd.read_csv(INPUT_CSV, encoding='utf-8')
df_all.columns = [col.strip().lower() for col in df_all.columns]

# Ensure 'lemme' is string
df_all['lemme'] = df_all['lemme'].astype(str)

# Working dataset will be reduced as we process
df_working = df_all.copy()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# chunk_num = 0
start_idx = 1

while True:
    # === STEP 2: TOP 400 LEMMES BY SPOKEN FREQUENCY (ISLEM == 1)
    spoken_candidates = (
        df_working[df_working['islem'] == 1]
        .drop_duplicates(subset='lemme')
        .sort_values(by='freqlemfilms', ascending=False)
    )

    if spoken_candidates.empty:
        break  # done

    top_spoken_lemmes = spoken_candidates['lemme'].tolist()[:SPOKEN_COUNT]

    # === STEP 3: COLLECT ALL ROWS FOR THESE LEMMES (regardless of islem)
    spoken_rows = pd.DataFrame()
    for lemme in top_spoken_lemmes:
        matches = df_working[df_working['lemme'] == lemme]
        spoken_rows = pd.concat([spoken_rows, matches], ignore_index=True)
        df_working = df_working[df_working['lemme'] != lemme]

    # === STEP 4: TOP 100 WRITTEN LEMMES BY freqlemlivres (ISLEM == 1)
    written_candidates = (
        df_working[df_working['islem'] == 1]
        .drop_duplicates(subset='lemme')
        .sort_values(by='freqlemlivres', ascending=False)
    )

    top_written_lemmes = written_candidates['lemme'].tolist()[:WRITTEN_COUNT]

    written_rows = pd.DataFrame()
    for lemme in top_written_lemmes:
        matches = df_working[df_working['lemme'] == lemme]
        written_rows = pd.concat([written_rows, matches], ignore_index=True)
        df_working = df_working[df_working['lemme'] != lemme]

    # === STEP 5: COMBINE AND EXPORT
    chunk_df = pd.concat([spoken_rows, written_rows], ignore_index=True)

    # calculate end index
    total_lemmes = len(set(chunk_df['lemme']))
    end_idx = start_idx + total_lemmes - 1

    # save file
    filename = f'Freq {start_idx} - {end_idx}.csv'
    filepath = os.path.join(OUTPUT_DIR, filename)
    chunk_df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"{filename}\tlen(set(chunk_df['lemme'])) = {total_lemmes} lemmes")

    # === PREPARE FOR NEXT CHUNK
    start_idx = end_idx + 1
    # chunk_num += 1

print("\nDone: All chunks generated.")
