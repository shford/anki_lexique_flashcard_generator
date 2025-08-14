"""
@Purpose: Ensures each export Anki file has exactly CHUNK_SIZE lemmes.

@Instructions:
  On the first one START should be run. If you run s3 again you should set START to whatever the
  highest filename number (number to the right of '-') plus 1.
"""
import os
import warnings

import pandas as pd

from s3_Export_Anki_Format import ANKI_CSV_OUTPUT_DIR as ANKI_CSV_DIR
from s3_Export_Anki_Format import CHUNK_SIZE
from s3_Export_Anki_Format import OUTPUT_PREFIX


def main():
    enforce_chunk_size_anki_cards()


def enforce_chunk_size_anki_cards():
    print(f'Enforcing flashcard deck size of {CHUNK_SIZE}...')

    # set up initial state
    df1 = None
    path2 = None
    df2 = None
    df_overflow = pd.DataFrame()

    # begin balancing number of rows
    NUM_FILES = len(os.listdir(ANKI_CSV_DIR))
    start = 1
    while NUM_FILES >= 1:
        filename_1 = f'{OUTPUT_PREFIX}{start} - {start + CHUNK_SIZE - 1}.csv'
        path1 = f'{ANKI_CSV_DIR}/{filename_1}'
        if df1 is None:
            df1 = pd.read_csv(path1, encoding='utf-8', header=None)

        if NUM_FILES >= 2:
            filename_2 = f'{OUTPUT_PREFIX}{start + CHUNK_SIZE} - {start + CHUNK_SIZE*2 - 1}.csv'
            path2 = f'{ANKI_CSV_DIR}/{filename_2}'
            try:
                df2 = pd.read_csv(path2, encoding='utf-8', header=None)
            except:
                # the next file DNE -> ignore error and normalize as much as
                warn_msg = 'WARNING: Likely running s4 without running s3 after increasing the number of DESIRED_FLASHCARDS.'
                warnings.warn(warn_msg)
                df2 = pd.DataFrame()
                NUM_FILES = 1
        elif NUM_FILES == 1:
            # on the last file there is no next file,
            # however to keep the balance logic working we'll make df2 an empty DataFrame
            df2 = pd.DataFrame()
            possible_overflow_filename = f'{OUTPUT_PREFIX}{start+CHUNK_SIZE} - {start + CHUNK_SIZE*2 - 1}.csv'
            possible_overflow_path = f'{ANKI_CSV_DIR}/{possible_overflow_filename}'
        else:
            raise FileNotFoundError('Cannot pass negative number of files.')

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
            if not df_overflow.empty:
                df_overflow.to_csv(possible_overflow_path)
                if possible_overflow_filename is None:
                    pass
                print(f'Wrote .csv: {possible_overflow_filename}')
        elif NUM_FILES == 2 and df2.empty and df_overflow.empty:
            # delete empty next file
            try:
                os.remove(path2)
                print(f'  Deleted empty file {filename_2}.')
            except:
                pass

            # write current file
            df1.to_csv(path1, index=False, encoding='utf-8', header=False)
            if filename_1 is None:
                pass
            print(f'Wrote .csv: {filename_1}')

            # no need for final run
            break
        elif NUM_FILES == 2 and df2.empty:
            # raise Exception - balance logic must be broken
            raise Exception("Wha' in tarnation? NUM_FILES==2 and df2.empty but df_overflow is not empty. Balance logic must be broken.")
        elif NUM_FILES == 2 and len(df2) <= CHUNK_SIZE and df_overflow.empty:
            # write non-empty df2
            df2.to_csv(path2, index=False, encoding='utf-8', header=False)
            if filename_2 is None:
                pass
            print(f'Wrote .csv: {filename_2}')

            # write current file
            df1.to_csv(path1, index=False, encoding='utf-8', header=False)
            if filename_1 is None:
                pass
            print(f'Wrote .csv: {filename_1}')

            # no need for final run
            break
        # endregion

        # write current file
        df1.to_csv(path1, index=False, encoding='utf-8', header=False)
        if filename_1 is None:
            pass
        print(f'Wrote .csv: {filename_1}')

        # set current to next
        start += CHUNK_SIZE
        df1 = df2

        # decrement loop
        NUM_FILES -= 1

    print(f'Enforced deck size.')
    print()
    return


# what if we get to df1 = 2nd to last file and df2 (last file) ends up empty?
if __name__ == '__main__':
    main()