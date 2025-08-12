"""
@Purpose: Mux spoken/written and write to file.

@Instructions:
  There's no need to run this directly as s3 will call it even if you don't. If you do run it,
    it won't hurt anything, it'll just write the muxed output csv.

  In terms of configuration...
    You can pretty much just decide you desired spoken/written ratio and
    number of desired Anki cards.

  For spoken to written ratio, the db we're generating these cards from labels words by
  how frequently they appear in different forms (movie subtitles would count toward spoken
  frequency, books/poems would count towards written).
    - If you want to focus more on academic french, I'd recommend
    increasing the proportion of WRITTEN_COUNT.
    - If you want to focus on speaking I'd recommend
    increasing the proportion of SPOKEN_COUNT.

    By default it's 400:100, so you'll get Anki decks
    of size 500 and each deck will the highest
    80% (400/(400+100)) spoken words and 20% (100/(400+100)) written words.

    You could do 100:0 (100%, 0%), or 475:25 (95%, 5%), or 0:300 (0%, 100%). You can even do really
    stupid decks like 7:2 which would result in generated deck sizes of 9.

    I'd recommend you choose the default deck size (derived from spoken + written)
    in relatively small increments and utilize nested subdecks to build a
    large study deck. This makes it nice so that when you go to add
    audio w/ HyperTTS, if you get rate limited by whatever plan for whatever provider
    you use, then you only wasted a relatively small # of API calls before your limit resets.
"""
# external libs
import pandas as pd

# project files
from s1_Filter_Lexique import OUTPUT_DIR
from s1_Filter_Lexique import OUTPUT_CSV as INPUT_CSV


# ==== CONFIGURATION ====
SPOKEN_COUNT = 475
WRITTEN_COUNT = 25
DESIRED_FLASHCARDS = 32500  # set equal to 0 for all available
OUTPUT_MUXED_CSV = f'{OUTPUT_DIR}/Lexique383 - Muxed.csv'
# ========================

# Derived Constants
CHUNK_SIZE = SPOKEN_COUNT + WRITTEN_COUNT
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
