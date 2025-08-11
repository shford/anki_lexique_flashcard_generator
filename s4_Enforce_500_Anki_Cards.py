"""
@Purpose: Ensures each export Anki file has exactly 500 lemmes.

@Instructions:
  On the first one START should be run. If you run s3 again you should set START to whatever the
  highest filename number (number to the right of '-') plus 1.

  If you don't do this then df_overflow will be prepend to the first file it sees and everything
  will be shifted down by that many rows.. which isn't a huge deal, it just means your frequencies will be
  slightly offset.
"""
import pandas as pd
import os

from s2_Mux_Lexique import DESIRED_FLASHCARDS as STOP
from s3_Export_Anki_Format import CHUNK_SIZE
from s3_Export_Anki_Format import OUTPUT_PREFIX, OVERFLOW_FILENAME, OVERFLOW_PATH
from s3_Export_Anki_Format import ANKI_CSV_OUTPUT_DIR as ANKI_CSV_DIR

# ==== Config Constants ====
START = 1
# START = 3001  # example: You create and balanced 1-3000 yesterday, it may it may have generated a df_overflow. You ran s3 today and now have 3001-6000. To rebalance 3001-6000 you would set this to 3001. Note: This program assumes anything in df_overflow at the start of the program is a higher frequency word and will prepend df_overflow and shift words as necessary. This is expected and desireable behavior.
# ========================

def main():
    global START
    check_indices(START, STOP)

    # set up initial state
    df1 = None
    filename2 = None
    path2 = None
    df2 = None
    df_overflow = None
    if os.path.exists(OVERFLOW_PATH):
        df_overflow = pd.read_csv(OVERFLOW_PATH, encoding='utf-8', header=None)
    else:
        df_overflow = pd.DataFrame()

    # begin balancing row count
    NUM_FILES = int((STOP - START + 1) / CHUNK_SIZE)
    while NUM_FILES >= 1:
        if df1 is None:
            # init df1 from CSV
            filename1 = f'{OUTPUT_PREFIX}{START} - {START + CHUNK_SIZE - 1}.csv'
            path1 = f'{ANKI_CSV_DIR}/{filename1}'
            try:
                df1 = pd.read_csv(path1, encoding='utf-8', header=None)
            except:
                print(f'No file "{filename1}" for starting lemme {START} found. Tip: Try adjusting starting lemme.')
                exit(0)
        else:
            # set current to next
            filename1 = filename2
            path1 = path2
            df1 = df2

        if NUM_FILES == 1:
            # on the last file there is no file2, however to keep the balance logic working we'll make df2 an empty DataFrame
            df2 = pd.DataFrame()
        else:
            # init df2 regardless of first or last run
            START += CHUNK_SIZE
            filename2 = f'{OUTPUT_PREFIX}{START} - {START + CHUNK_SIZE - 1}.csv'
            path2 = f'{ANKI_CSV_DIR}/{filename2}'
            try:
                df2 = pd.read_csv(path2, encoding='utf-8', header=None)
            except:
                df2 = pd.DataFrame()
                NUM_FILES = 1 # clearly the next file DNE (the files < DESIRED_CARDS but that's fine)

        # prepend from df_overflow prior to balancing regardless of length
        if len(df_overflow) > 0:
            df1 = pd.concat([df_overflow.iloc[:len(df_overflow)], df1], ignore_index=True)
            df_overflow = df_overflow.drop(df_overflow.index[:len(df_overflow)]).reset_index(drop=True)

        # region balance rows
        difference = len(df1) - CHUNK_SIZE
        if difference > 0:
            # append leftover from df1 to df_overflow
            excesss_df1 = df1.iloc[CHUNK_SIZE:CHUNK_SIZE + difference]
            df_overflow = pd.concat([df_overflow, excesss_df1])
            df1 = df1.iloc[:-difference]
        elif difference == 0:
            pass  # noop
        elif difference < 0:
            # append to df1 from df2
            difference = abs(difference)
            df1 = pd.concat([df1, df2.iloc[:difference]])

            # remove from df2
            df2 = df2.drop(df2.index[:difference])
        # endregion

        # region save ODS documents
        if NUM_FILES == 1:
            # delete overflow if unneeded
            if df_overflow.empty and os.path.exists(OVERFLOW_PATH):
                os.remove(OVERFLOW_PATH)
            else:
                df_overflow.to_csv(OVERFLOW_PATH, index=False, encoding='utf-8', header=False)
                print('  Saved overflow file for next run.')

            # write current file
            df1.to_csv(path1, index=False, encoding='utf-8', header=False)
            print(f'Wrote final .csv: {filename1}')
        elif NUM_FILES == 2 and df2.empty and df_overflow.empty:
            # delete empty overflow file
            if os.path.exists(OVERFLOW_PATH):
                os.remove(OVERFLOW_PATH)
                print(f'  Deleted empty file {OVERFLOW_FILENAME}.')

            # delete empty next file
            os.remove(path2)
            print(f'  Deleted empty file {filename2}.')

            # write current file
            df1.to_csv(path1, index=False, encoding='utf-8', header=False)
            print(f'Wrote final .csv: {filename1}')

            # no need for final run
            break
        elif NUM_FILES == 2 and df2.empty:
            # raise Exception - balance logic must be broken
            raise Exception(
                "Wha' in tarnation? NUM_FILES==2 and df2.empty but df_overflow is not empty. Balance logic must be broken.")
        elif NUM_FILES == 2 and len(df2) <= 500 and df_overflow.empty:
            # delete empty overflow file
            if os.path.exists(OVERFLOW_PATH):
                os.remove(OVERFLOW_PATH)
                print(f'Deleted empty file {OVERFLOW_FILENAME}.')

            # write df1
            df1.to_csv(path1, index=False, encoding='utf-8', header=False)
            print(f'Wrote: {filename1}')

            # write non-empty df2
            df2.to_csv(path2, index=False, encoding='utf-8', header=False)
            print(f'Wrote final .csv: {filename2}')

            # no need for final run
            break
        else:
            # write df1 to .csv file
            df1.to_csv(path1, index=False, encoding='utf-8', header=False)
            print(f'Wrote: {filename1}')
        # endregion

        # decrement loop
        NUM_FILES -= 1


def check_indices(start, stop):
    if (stop - start + 1) < 1000:
        raise Exception('START and STOP are insufficently spaced to open two files.')

    if (start - 1) % 500 != 0:
        raise Exception('Invalid START - ensure it ends in 1.')

    if stop % 500 != 0:
        raise Exception('Invalid STOP - ensure it is divisible by 500.')

    return


# what if we get to df1 = 2nd to last file and df2 (last file) ends up empty?
if __name__ == '__main__':
    main()