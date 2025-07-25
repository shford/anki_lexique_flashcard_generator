"""
@Purpose: Ensures each export Anki file has exactly 500 lemmes.

This... shouldn't exist and is horrible programming. I ought to just correctly
 calculate the number of rows in the initial exports in file 2. or 3. or just combine
 all the files into one large program that reads efficiently in chunks.
"""
import pandas as pd
import os

from s3_Export_Anki_Format import CHUNK_SIZE
from s3_Export_Anki_Format import OUTPUT_PREFIX
from s3_Export_Anki_Format import ANKI_CSV_OUTPUT_DIR as ANKI_CSV_DIR

# ==== Configuration ====
# see previous steps
# ========================

def main():
    START = 1
    STOP = 1000

    check_indices(START, STOP)

    # set up initial state
    df1 = None
    filename2 = None
    path2 = None
    df2 = None
    df_overflow = None
    overflow_filename = f'df_overflow.csv'
    overflow_path = f'{ANKI_CSV_DIR}/{overflow_filename}'
    if os.path.exists(overflow_path):
        df_overflow = pd.read_csv(overflow_path, encoding='utf-8', header=None)
    else:
        df_overflow = pd.DataFrame()

    # begin balancing row count
    NUM_FILES = int((STOP-START+1)/CHUNK_SIZE)
    while NUM_FILES >= 1:
        if df1 is None:
            # init df1 from CSV
            filename1 = f'anki_deck_{START} - {START+CHUNK_SIZE-1}.csv'
            path1 = f'{ANKI_CSV_DIR}/{filename1}'
            df1 = pd.read_csv(path1, encoding='utf-8', header=None)
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
            filename2 = f'{OUTPUT_PREFIX}{START} - {START+CHUNK_SIZE-1}.csv'
            path2 = f'{ANKI_CSV_DIR}/{filename2}'
            df2 = pd.read_csv(path2, encoding='utf-8', header=None)


        # prepend from df_overflow prior to balancing regardless of length
        if len(df_overflow) > 0:
            df1 = pd.concat([df_overflow.iloc[:len(df_overflow)], df1], ignore_index=True)
            df_overflow = df_overflow.drop(df_overflow.index[:len(df_overflow)]).reset_index(drop=True)

        # region balance rows
        difference = len(df1) - CHUNK_SIZE
        if difference > 0:
            # append leftover from df1 to df_overflow
            excesss_df1 = df1.iloc[CHUNK_SIZE:CHUNK_SIZE+difference]
            df_overflow = pd.concat([df_overflow, excesss_df1])
            df1 = df1.iloc[:-difference]
        elif difference == 0:
            pass # noop
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
            if df_overflow.empty and os.path.exists(overflow_path):
                os.remove(overflow_path)
            else:
                df_overflow.to_csv(overflow_path, index=False, encoding='utf-8', header=False)
                print('  Saved overflow file for next run.')

            # write current file
            df1.to_csv(path1, index=False, encoding='utf-8', header=False)
            print(f'Wrote final .csv: {filename1}')
        elif NUM_FILES == 2 and df2.empty and df_overflow.empty:
            # delete empty overflow file
            if os.path.exists(overflow_path):
                os.remove(overflow_path)
                print(f'  Deleted empty file {overflow_filename}.')

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
            if os.path.exists(overflow_path):
                os.remove(overflow_path)
                print(f'Deleted empty file {overflow_filename}.')

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
main()
