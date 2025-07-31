"""
@Purpose: Mux spoken/written and write to file.
"""
# external libs
import pandas as pd

# project files
from s1_Filter_Lexique import OUTPUT_DIR
from s1_Filter_Lexique import OUTPUT_CSV as INPUT_CSV


# ==== CONFIGURATION ====
SPOKEN_COUNT = 400
WRITTEN_COUNT = 100
MUX_CHUNK_SIZE = SPOKEN_COUNT + WRITTEN_COUNT
DESIRED_FLASHCARDS = 20000  # set equal to 0 for all available
OUTPUT_MUXED_CSV = f'{OUTPUT_DIR}/Lexique383 - Muxed.csv'
# ========================

# Derived Constants
SELECT_N_LEMMES = int(DESIRED_FLASHCARDS * 1.25) # apply fuzz factor


def main():
    muxed_lemmes, muxed_df = mux_frequencies()
    muxed_df.to_csv(OUTPUT_MUXED_CSV, index=False, encoding='utf-8')
    print(f'\nDone: Wrote df w/ {len(muxed_lemmes)} intermixed by frequencies of spoken/written.')


def mux_frequencies() -> (list, pd.DataFrame):
    # load csv
    df_all = pd.read_csv(INPUT_CSV, encoding='utf-8')

    # working dataset will be reduced as we process
    df_working = df_all.copy()

    muxed_lemmes = []
    muxed_df = pd.DataFrame()
    while len(muxed_lemmes) < SELECT_N_LEMMES or SELECT_N_LEMMES == 0:
        # top lemmes by spoken and frequency
        highest_spoken = get_top_lemmes(df_working, SPOKEN_COUNT, 'freqlemfilms')
        df_working = df_working[~df_working['lemme'].isin(highest_spoken)]

        highest_written = get_top_lemmes(df_working, WRITTEN_COUNT, 'freqlemlivres')
        df_working = df_working[~df_working['lemme'].isin(highest_written)]

        top_lemmes = highest_spoken + highest_written # combine lists
        lemme_df = df_all[df_all['lemme'].isin(top_lemmes)].copy()
        lemme_df.loc[:, '__order'] = pd.Categorical(lemme_df['lemme'], categories=top_lemmes, ordered=True)
        lemme_df = lemme_df.sort_values('__order').drop(columns='__order')

        muxed_lemmes.extend(top_lemmes)
        muxed_df = pd.concat([muxed_df, lemme_df],ignore_index=True)

        if not highest_spoken and not highest_written:
            break  # exhausted candidates

    return muxed_lemmes, muxed_df


def get_top_lemmes(df, count, freq_col_name) -> list:
    return (
        df[df['islem'] == 1]
        .drop_duplicates(subset='lemme')
        .sort_values(by=freq_col_name, ascending=False)['lemme']
        .tolist()[:count]
    )


if __name__ == "__main__":
    main()
