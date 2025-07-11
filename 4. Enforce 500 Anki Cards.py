"""
@Purpose: Ensures each export Anki file has exactly 500 lemmes.

This... shouldn't exist and is horrible programming. I ought to just correctly
 calculate the number of rows in the initial exports in file 2. or 3. or just combine
 all the files into one large program that reads efficiently in chunks.
"""
import pandas as pd
import os


# ==== Configuration ====
USER_PATH = os.path.expanduser('~')
DIR = f'{USER_PATH}/Documents/flashcard_project_new/anki_lexique_imports'
OUTPUT_PREFIX = 'anki_deck_'
CHUNK_SIZE = 500
# ========================

def main():
    START = 1
    STOP = 1500

    check_indices(START, STOP)

    # set up initial state
    df1 = None
    filename2 = None
    path2 = None
    df2 = None
    df_overflow = None
    overflow_filename = f'df_overflow.csv'
    overflow_path = f'{DIR}/{overflow_filename}'
    if os.path.exists(overflow_path):
        df_overflow = pd.read_csv(overflow_path, encoding='utf-8')
    else:
        df_overflow = pd.DataFrame()

    # begin balancing row count
    NUM_FILES = int((STOP-START+1)/CHUNK_SIZE)
    while NUM_FILES >= 1:
        if df1 is None:
            # init df1
            filename1 = f'anki_deck_{START} - {START+CHUNK_SIZE-1}.csv'
            path1 = f'{DIR}/{filename1}'
            df1 = pd.read_csv(path1, encoding='utf-8')
        else:
            filename1 = filename2
            path1 = path2
            df1 = df2

        if NUM_FILES == 1:
            # on the last file there is not file2, however to keep the balance logic working we'll make df2 an empty DataFrame
            df2 = pd.DataFrame()
        else:
            # init df2 regardless of first or last run
            START += CHUNK_SIZE
            filename2 = f'anki_deck_{START} - {START+CHUNK_SIZE-1}.csv'
            path2 = f'{DIR}/{filename2}'
            df2 = pd.read_csv(path2, encoding='utf-8')

        # balance
        difference = len(df1) - CHUNK_SIZE
        if difference > 0:
            # save leftover from df1 to df_overflow
            excesss_df1 = df1.iloc[CHUNK_SIZE:CHUNK_SIZE+difference]
            df_overflow = pd.concat([df_overflow, excesss_df1])
            df1 = df1.iloc[:-difference]
        elif difference == 0:
            pass # noop
        elif difference < 0:
            # check if df_overflow has enough to take from
            if len(df_overflow) >= abs(difference):
                # move difference from df_overflow to df1 and then remove from df_overflow
                df1 = df1.merge(df_overflow.iloc[:abs(difference)], how='left')
                df_overflow = df_overflow.drop(df_overflow.index[:abs(difference)])
            else:
                # empty df_overflow and update remaining difference
                if not df_overflow.empty:
                    len_df_overflow = len(df_overflow)
                    df1 = df1.merge(df_overflow.iloc[:len_df_overflow], how='left')
                    df_overflow = df_overflow.drop(df_overflow.index[:len_df_overflow])
                    difference += len_df_overflow

                len_df2 = len(df2)
                if abs(difference) < len_df2:
                    # take the deficit from df2 - no need to update difference b/c it'll be 0
                    df1 = df1.merge(df2.iloc[:difference], how='left')
                    df2 = df2.drop(df2.index[:difference])
                else:
                    # note: this will be the last file  - need to end one early - write df1, remove df2 & overflow - no need to update difference b/c we've done all we can do
                    df1 = df1.merge(df2.iloc[:len_df2], how='left')
                    df2 = df2.drop(df2.index[:len_df2])

        # save ODS documents
        if NUM_FILES == 1:
            # delete overflow if unneeded
            if df_overflow.empty:
                os.remove(overflow_path)
            else:
                df_overflow.to_csv(overflow_path, index=False, encoding='utf-8')
                print('  Saved overflow file for next run.')

            # write current file
            df1.to_csv(path1, index=False, encoding='utf-8')
            print(f'Wrote final .csv: {filename1}')
        elif NUM_FILES == 2 and df2.empty and df_overflow.empty:
            # delete empty overflow file
            os.remove(overflow_path)
            print(f'  Deleted empty file {overflow_filename}.')

            # delete empty next file
            os.remove(path2)
            print(f'  Deleted empty file {filename2}.')

            # write current file
            df1.to_csv(path1, index=False, encoding='utf-8')
            print(f'Wrote final .csv: {filename1}')

            # no need for final run
            break
        elif NUM_FILES == 2 and df2.empty:
            raise Exception("Wha' in tarnation? NUM_FILES==2 and df2.empty but df_overflow is not empty.")
        elif NUM_FILES == 2 and df_overflow.empty:
            # delete empty overflow file
            os.remove(overflow_path)
            print(f'  Deleted empty file {overflow_filename}.')

            # write df1
            df1.to_csv(path1, index=False, encoding='utf-8')
            print(f'Wrote: {filename1}')

            # write non-empty df2
            df2.to_csv(path2, index=False, encoding='utf-8')
            print(f'Wrote final .csv: {filename2}')

            # no need for final run
            break
        else:
            # write df1 to .csv file
            df1.to_csv(path1, index=False, encoding='utf-8')
            print(f'Wrote: {filename1}')

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
