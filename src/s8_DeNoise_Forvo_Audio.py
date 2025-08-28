"""
@Purpose:       Provide a variety of free denoising options.

@Instructions:  **It is unlikely that you will need to change anything in this file.**
                **Just run it exactly once. It will likely take a while, possibly hours.**
                      Repeated runs data may (read as: will) cause degraded sound quality.

Warning:        This will overwrite the old files sound files. However, by default all files will be
                backed up in collection.media.backup_{timestamp}. If you prefer to disable this
                option, toggle BACKUP=True to BACKUP=False (NOT RECOMMENDED).

                To revert unwanted audio changes, simply delete collection.media and rename the .backup_{timestamp}
                folder to collection.media

A note about ffmpeg-normalize:
    I appreciate that it exists. That said, it's not ffmpeg; it's a wrapper. It seems to work best with .wav files.
    For some reason the official documentation is lacking and the latest github and pypi pages have no
    useful info.
    -
    However, (the old) ffmpeg-normalize version 1.16.0 actually had really good examples and is immortalized on pypi.
    https://pypi.org/project/ffmpeg-normalize/1.16.0/
"""
from ast import literal_eval
import datetime
import io
import os
import shutil
import subprocess
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import librosa
# neural nine imports
import numpy as np
import soundfile as sf
from scipy.signal import medfilt

from s6_Import_Packages_Into_Anki import PROFILE


# ==== Configuration ====
BACKUP = True                       # recommend True
CHECK_FOR_CORRUPT_FILES = True      # recommend True
RESTORE_CORRUPT_FILES = True        # recommend True; works only if CHECK_FOR_CORRUPT_FILES is True
WRITE_INTERMEDIATE_FILES = False    # recommend False; True is helpful if you want to manually fine tune what each filter functions
SELECTED_AUDIO_PREFIX = 'hypertts'  # recommend leave as hypertts for Forvo; otherwise open card and see what your generated file names looks like
ALT_TXT_IN_AUDIO = '_fr_'           # recommend leave as is; just happens to constant string that works
# ========================

# Formatting Constants - strongly recommend you do not change DESIRED_
WRITE_RATE = '44100'
WRITE_CHANNELS = '1'
WRITE_FORMAT = 'mp3'
WRITE_CODEC = 'libmp3lame'
DESIRED_RATE = '44100'
DESIRED_CHANNELS = '1'
DESIRED_FORMAT = 's16le'
DESIRED_CODEC = 'pcm_s16le'
BYTES_PER_SAMPLE = '2'  # 16-bit = 2 bytes

# Intermediate File Naming Constants (for debugging/fine tuning ffmpeg arguments)
USER_PATH = os.path.expanduser('~')
ANKI_DIR = f'{USER_PATH}/.local/share/Anki2/{PROFILE}/collection.media'
PROJECT_NAME = '.anki_lexique_flashcard_generator'
NEURAL_NINE_PREFIX = 'n9stft_'
SILENCE_RM_PREFIX='silence_rm_'
ARNNDN_PREFIX = 'arnndn_'
AFFTDN_PREFIX = 'afftdn_'
LHPASS_PREFIX = 'lhpass_'
ADECLICK_PREFIX = 'adeclick_'
AFWTDN_PREFIX = 'afwtdn_broadband_'
ANLMDN_PREFIX = 'anlmdn_broadband_least_'
ADYNAMICEQ_PREFIX = 'adynamiceq_'
EXTERNAL_FFMPEG_NORMALIZE = 'extern_ffmpeg_norm_'
SPEECHNORM_PREFIX = 'speechnorm_'
NORMALIZE_PREFIX = 'norm_'
EQUALIZER_PREFIX = 'final_'


def main():
    override_prog_configs_from_file(globals())
    corrupt_files_prior = set()
    corrupt_files_after = set()

    t1 = time.time()
    if BACKUP:
        print('Backing up collection.')
        backup_audio_collection()

    # populate filenames
    dir_contents = os.listdir(ANKI_DIR)
    hypertts_mp3_filenames = [f for f in dir_contents if (os.path.isfile(os.path.join(ANKI_DIR, f)) and SELECTED_AUDIO_PREFIX in f)]
    std_forvo_api_mp3_filenames = [f for f in dir_contents if (os.path.isfile(os.path.join(ANKI_DIR, f)) and ALT_TXT_IN_AUDIO in f and WRITE_FORMAT in f and not 'ATTS ' in f)]
    mp3_filenames = hypertts_mp3_filenames + std_forvo_api_mp3_filenames

    if CHECK_FOR_CORRUPT_FILES:
        print('Checking for corrupt audio files. This may take some time...')
        corrupt_files_prior = get_num_corrupt_audio_files_and_attempt_restore(mp3_filenames, corrupt_files_prior)
        print(f'Prior to running, found {len(corrupt_files_prior)} corrupt audio files in directory:\n{ANKI_DIR}.\n')

    # parallelize
    print(f'Executing denoising program on {len(mp3_filenames)} files.\n')
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_audio_file, mp3_filenames)
    # old serial processing for easy debugging/profiling
    # for filename in mp3_filenames:
    #     process_audio_file(filename)

    if CHECK_FOR_CORRUPT_FILES:
        corrupt_files_after = get_num_corrupt_audio_files_and_attempt_restore(mp3_filenames, corrupt_files_after)
        newly_corrupted = corrupt_files_after - corrupt_files_prior
        if len(newly_corrupted) > 0:
            print('\nDetected the following audio files were corrupted during runtime:')
            [print(c) for c in newly_corrupted]
            print(f'Restoration failed. Recommend manual restore from most recent backup:\n\t{USER_PATH}/.local/share/Anki2/{PROFILE}/\n')
            print()

    t2 = time.time()
    print(f'\nWrote {len(mp3_filenames)} new files in {t2-t1} seconds.\n.')


def override_prog_configs_from_file(global_symbol_table) -> None:
    """
    This function exists mostly for the author's edification. You probably don't need a
    config file.

    I kept forgetting to toggle the default values back to what should be the default so
    now we're just going to override them from a local config script.

    I can't git skip tree on just part of a file so... here we are.

    If you really feel like you need a config file the format is:
    {
        'BACKUP': 'True',
        'SomeOtherGlobal': '5',
        'Savvy?': 'Aye',
    }
    :return:
    """
    config_path = '../resources/config.txt'

    # ensure file exists, make file template if it doesn't
    if not os.path.exists(config_path):
        return

    with open(config_path, 'r') as f:
        config_raw = f.read()

        try:
            config_globals_dict = literal_eval(config_raw)

            for key in config_globals_dict.keys():
                if not key == 'comment': # skip comments
                    # only modify if global exists
                    if key not in global_symbol_table:
                        continue

                    # ensure we're importing settings that make sense
                    imported_global_value = literal_eval(config_globals_dict[key])
                    if type(global_symbol_table[key]) != type(imported_global_value):
                        raise ValueError

                    # assign global
                    global_symbol_table[key] = literal_eval(config_globals_dict[key])
        except SyntaxError or ValueError as e:
            print(
                f'\nCredential file at {config_path} is malformed.\nNote: if you delete your file and re-run this program it will remake a sane template.')
            exit(-1)


def backup_audio_collection():
    t = datetime.datetime.now()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    src_collection = f'{USER_PATH}/.local/share/Anki2/{PROFILE}/collection.media'
    dst_collection = f'{USER_PATH}/.local/share/Anki2/{PROFILE}/collection.media.backup_{timestamp}'
    shutil.copytree(src_collection, dst_collection, dirs_exist_ok=True) # overwriting is a-okay :)


def get_num_corrupt_audio_files_and_attempt_restore(filenames: list, corrupt_files: set):
    bad_files = []
    for filename in filenames:
        path = f'{ANKI_DIR}/{filename}'

        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', path, '-f', 'null', '-'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stderr = result.stderr.strip()
        if stderr:
            # print(f"ffmpeg stderr for {path}:\n{stderr}\n") # log stderr
            # flag if there are serious decoding errors
            if "Invalid" in stderr or "error" in stderr.lower():
                print(f'   Found corrupt audio file: {filename}')
                corrupt_files.add(filename)

                # attempts to restore corrupt files from most recent backup
                if RESTORE_CORRUPT_FILES:
                    backup_dir = get_newest_backup_dir(f'{ANKI_DIR}/..')
                    if backup_dir is not None:
                        try:
                            shutil.copy(f'{backup_dir}/{filename}', path)
                            print(f'   Restored corrupt file from backup.')
                            print('')
                            corrupt_files.remove(filename)
                        except:
                            print(f'   Failed to restore corrupt file from backup.')
                            print('')

    return corrupt_files


def get_newest_backup_dir(backup_parent_dir, restore_from_oldest_backup=False):
    dirs = [d for d in Path(backup_parent_dir).iterdir() if d.is_dir() and 'collection.media.backup_' in d.name]
    if len(dirs) == 0:
        return None

    if restore_from_oldest_backup:
        return min(dirs, key=lambda d: d.stat().st_mtime, default=None)
    else:
        return max(dirs, key=lambda d: d.stat().st_mtime, default=None)


def process_audio_file(filename):
    # read audio file into required format
    filepath = f'{ANKI_DIR}/{filename}'
    pcm_bytes = _read_audiofile_to_pcm_bytes(filepath) # note len is in bytes
    if pcm_bytes is None or len(pcm_bytes) < 1024: # too short files are most likely malformed
        return # don't continue processing corrupt files

    # file is not corrupt, but is most likely AI generated, so no
    # filtering necessary. these sizes were determined by manually
    # examining file sizes of azure voices. From what I saw, no
    # recorded voices from ~30k Forvo clips fall within this size range
    # and all AI voices did.
    if (1024 * 16 + 1) < len(pcm_bytes) < (1024 * 30.8 + 1):
        return

    # clean up audio using ffmpeg wrappers
    pcm_bytes = audio_chain(pcm_bytes, filename)

    # write audio to mp3 file
    mp3_data = _convert_pcm_to_mp3(pcm_bytes)
    with open(filepath, 'wb') as f:
        f.write(mp3_data)
    print(f'Wrote: {filename}.')


def audio_chain(pcm_bytes, filename):
    # 1. remove silence
    pcm_bytes = ffmpeg_silenceremove(pcm_bytes, filename)

    # 2. remove non-speech (coughing, sniffling, shuffling, humming, etc) - note: model works best early in process
    sr_model_path = '../resources/rnnoise_models/speech_recording.rnnn'
    secondary_prefix = '_sr_'
    pcm_bytes = ffmpeg_arnndn(pcm_bytes, filename, sr_model_path, secondary_prefix)

    # 3. high/low pass filter helps with extreme noises
    pcm_bytes = ffmpeg_lowpass_highpass(pcm_bytes, filename)

    # 4. x3 rounds of afftdn for general clarity
    # pcm_bytes = ffmpeg_afftdn(pcm_bytes, filename)
    # pcm_bytes = ffmpeg_afftdn(pcm_bytes, filename)
    # pcm_bytes = ffmpeg_afftdn(pcm_bytes, filename)

    # 5. stft is good at cleaning front and tail
    # pcm_bytes = neural_nine_demo(pcm_bytes, filename)

    # skip local volume normalization - it ruins emphasis

    # 6. global volume normalization
    step6_bytes = external_tool_ffmpeg_normalize(pcm_bytes, filename)

    # 7. rebrighten a bit
    pcm_bytes = equalizer(step6_bytes, filename)
    return pcm_bytes


def wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, prefix=None):
    """
    run ffmpeg_commands
    """
    if WRITE_INTERMEDIATE_FILES and prefix is None:
        raise Exception('Error: Missing argument to run_command. If specifying write behavior, must pass filename to write to.')

    try:
        result = subprocess.run(command, input=pcm_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        pcm_bytes = result.stdout
        if WRITE_INTERMEDIATE_FILES:
            intermediate_file_path = f'{ANKI_DIR}/{prefix}_{filename}'
            with open(intermediate_file_path, 'wb') as f:
                f.write(_convert_pcm_to_mp3(pcm_bytes))

        return pcm_bytes
    except subprocess.CalledProcessError as e:
        print('Stderr:\n')
        print(e.stderr.decode('utf-8'))
        raise


def wrap_subprocess_run(command, optional_input=None):
    try:
        if optional_input is None:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        else:
            result = subprocess.run(command, input=optional_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print('Stderr:\n')
        print(e.stderr.decode('utf-8'))
        return None


# =============================================
# ffmpeg silence removal - head/tail of audio
# =============================================
def ffmpeg_silenceremove(pcm_bytes, filename):
    """
    Remove silence from head and tail of audio stream.
    """
    command = [
    'ffmpeg',
    '-f', DESIRED_FORMAT,       # unchanging input format
    '-ar', DESIRED_RATE,        # unchanging input rate
    '-ac', DESIRED_CHANNELS,    # unchanging input channels
    '-i', 'pipe:0',             # unchanging stdin
    '-filter:a', 'silenceremove=start_periods=1:start_threshold=0:stop_periods=1:stop_threshold=0:detection=rms',
    '-f', DESIRED_FORMAT,       # unchanging output format
    'pipe:1'                    # unchaning stdout
    ]
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, SILENCE_RM_PREFIX)

# =============================================
# region ffmpeg audio filters
# =============================================
def denoise_examples():
    filename = 'hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3' # sample is: 'aimer'
    filepath = f'{ANKI_DIR}/{filename}'
    pcm_bytes = _read_audiofile_to_pcm_bytes(filepath)

    # short time fourier transform filtering from youtuber 'Neural Nine'
    pcm_bytes = neural_nine_demo(pcm_bytes, filename)

    # region neural network rnnoise models
    model_path = '../resources/rnnoise_models'

    gg_secondary_prefix = '_gg_'
    gg_model = f'{model_path}/general_general.rnnn'
    pcm_bytes = ffmpeg_arnndn(pcm_bytes, filename, gg_model, gg_secondary_prefix)

    gr_secondary_prefix = '_gr_'
    gr_model = f'{model_path}/general_recording.rnnn'
    pcm_bytes = ffmpeg_arnndn(pcm_bytes, filename, gr_model, gr_secondary_prefix)

    vg_secondary_prefix = '_vg_'
    vg_model = f'{model_path}/voice_general.rnnn'
    pcm_bytes = ffmpeg_arnndn(pcm_bytes, filename, vg_model, vg_secondary_prefix)

    vr_secondary_prefix = '_vr_'
    vr_model = f'{model_path}/voice_recording.rnnn'
    pcm_bytes = ffmpeg_arnndn(pcm_bytes, filename, vr_model, vr_secondary_prefix)

    sr_secondary_prefix = '_sr_'
    sr_model = f'{model_path}/speech_recording.rnnn'
    pcm_bytes = ffmpeg_arnndn(pcm_bytes, filename, sr_model, sr_secondary_prefix)
    # endregion

    # high/low pass (o.k. bandpass) filter helps with extreme noises
    pcm_bytes = ffmpeg_lowpass_highpass(pcm_bytes, filename)

    # afftdn for general clarity
    pcm_bytes = ffmpeg_afftdn(pcm_bytes, filename)

    # adeclick - impulsive filtering
    ffmpeg_adeclick(filepath, filename)

    # afwtdn - broadband filtering - I found this one to be uniquely bad. Maybe I just didn't tweak it enough.
    ffmpeg_afwtdn(pcm_bytes, filepath)

    # anldmn - broadband filtering (ref UCLA math article/heat equation)
    ffmpeg_anlmdn(pcm_bytes, filename)

    # adynamicequalizer - attenuate unwanted freqs
    ffmpeg_adynamicequalizer(pcm_bytes, filename)


def neural_nine_demo(pcm_bytes, filename):
    """
    FFT denoising written by NeuralNine (youtuber).
    Seems to be geared toward broadband filtering rather than impulse.
    """
    # load from audio file
    # y, sr = librosa.load(path_in, sr=None)

    # make numpy audio
    y = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sr = int(DESIRED_RATE)

    # process y,sr
    S_full, phase = librosa.magphase(librosa.stft(y))
    noise_power = np.mean(S_full[:, :int(sr*0.1)], axis=1)
    mask = S_full > noise_power[:, None]
    mask = mask.astype(float)
    k = max(1, min(5, mask.shape[1]) // 2 * 2 + 1 if mask.shape[1] >= 1 else 1)
    mask = medfilt(mask, kernel_size=(1,k))
    S_clean = S_full * mask
    y_clean = librosa.istft(S_clean * phase)

    # write intermediate files
    if WRITE_INTERMEDIATE_FILES:
        path_out = f'{ANKI_DIR}/{NEURAL_NINE_PREFIX}_{filename}'
        sf.write(path_out, y_clean, sr)

    # convert to pcm_bytes, sampling_rate prior to formatting
    buffer = io.BytesIO()
    sf.write(buffer, y_clean, sr, format='RAW', subtype='PCM_16')
    buffer_pcm_bytes, sampling_rate = buffer.getvalue(), str(sr)

    # ensure proper ffmpeg format
    command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,  # unchanging input format
        '-acodec', DESIRED_CODEC,
        '-ar', DESIRED_RATE,  # unchanging input rate
        '-ac', DESIRED_CHANNELS,  # unchanging input channels
        '-i', 'pipe:0',  # unchanging stdin
        '-f', DESIRED_FORMAT,  # unchanging output format
        '-acodec', DESIRED_CODEC,
        'pipe:1'  # unchaning stdout
    ]
    output = wrap_subprocess_run(command, buffer_pcm_bytes) # output is result.out or None
    return output # pcm bytes in exactly our required format


def ffmpeg_arnndn(pcm_bytes, filename, rel_model_path, secondary_prefix, mix=1):
    """
    arnndn params:
        pcm_bytes   : rnnoise model operates only on RAW 16-bit (machine endian) mono - e.g. pcm_bytes
        filename    : original audio file name
        model       : the model file (.rnnn)
        mix         : 0 is original, 1 is filtered, (0-1) is a blend of original and filtered, negative values are what was filtered out rather than filtered for (e.g. -1 is just noise)

    Equivalent cli command:
        ffmpeg -i <input_file.mp3> -filter:a arnndn=model=<model_path.rnnn>:mix
    """
    # fix model not found in subprocess calls
    abs_model_path = os.path.abspath(rel_model_path)
    command = [
    'ffmpeg',
    '-f', DESIRED_FORMAT,       # unchanging input format
    '-ar', DESIRED_RATE,        # unchanging input rate
    '-ac', DESIRED_CHANNELS,    # unchanging input channels
    '-i', 'pipe:0',             # unchanging stdin
    '-filter:a', f'arnndn=model={abs_model_path}:mix={mix}',
    '-f', DESIRED_FORMAT,       # unchanging output format
    'pipe:1'                    # unchaning stdout
    ]
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, f'{ARNNDN_PREFIX}{secondary_prefix}')


def _read_audiofile_to_pcm_bytes(input_path) -> (bytes|None):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-f', DESIRED_FORMAT,
        '-acodec', DESIRED_CODEC,
        '-ac', DESIRED_CHANNELS,
        '-ar', DESIRED_RATE,
        'pipe:1'  # stdout
    ]

    output = wrap_subprocess_run(command) # output is result.out or None
    if output is None:
        return None
    return output


def _convert_pcm_to_mp3(pcm_data):
    command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-acodec', DESIRED_CODEC,
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-f', WRITE_FORMAT,         # mp3 format
        '-acodec', WRITE_CODEC,     # mp3 codec
        '-ar', WRITE_RATE,          # mp3 rate (same as normal)
        '-ac', WRITE_CHANNELS,      # mp3 channels (same as normal)
        '-q:a', '3',                # save space
        'pipe:1'                    # unchanging stdout
    ]
    result = subprocess.run(command, input=pcm_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout


def ffmpeg_afftdn(pcm_bytes, filename):
    """
    Sample frequencies referenced from Matteo M. on S.O.

    reference command:
    ffmpeg -i <path_in> -af "afftdn=nf=-25" <path_out>
    """
    command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'afftdn=nf=-25',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, AFFTDN_PREFIX)


def ffmpeg_lowpass_highpass(pcm_bytes, filename):
    """
    Sample frequencies referenced from Matteo M. on S.O.

    reference command:
    ffmpeg -i <path_in> -af "lowpass=f=3000, highpass=f=200" <path_out>
    """
    command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', f'highpass=f=100, lowpass=f=3000',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, LHPASS_PREFIX)


def ffmpeg_adeclick(pcm_bytes, filename):
    """
    reference command:
    ffmpeg -y -i <path_in> -af 'adeclick=w=60:o=85:a=10:t=3:b=5:m=s'
     -acodec libmp3lame -ar 48000 -ac 1 <path_out>

    Filter explanation (gist: uses slightly more aggressive noise removal)
    -af 'adeclick=w=60:o=85:a=10:t=3:b=5:m=s'
         │   │   │   │   │   └── overlap-save method                    [a or s]
         │   │   │   │   └───── burst fusion = 5% of window             [0, 2]
         │   │   │   └───────── threshold = 3                           [1, 100]
         │   │   └───────────── autoregression order = 10% of window    [0, 25]
         │   └───────────────── overlap = 85%                           [50, 95]
         └───────────────────── window size = 60 ms                     [10, 100]
    """
    command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'adeclick=w=10:o=95:a=0:t=1:b=0:m=s',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, ADECLICK_PREFIX)


def ffmpeg_afwtdn(pcm_bytes, filename):
    """
    todo
        - More of a weird note really, the default and mild commands both SIGSEGV's. The issue must
          presumably be in the underlying ffmpeg C/C++ files. I tried several variations of the arguments.
          I'm pretty sure I narrowed the issues down to the wavets: sym2 & sym4.
           Installed version at the time of writing was ffmpeg version 6.1.1-3ubuntu5 on Ubuntu 24.04.3.

    reference command:
    ffmpeg -y -i <path_in>
     -af 'afwtdn=sigma=0:levels=10:wavet=sym2:percent=85:profile=0:adaptive=0:samples=8192:softness=1'
     -acodec libmp3lame -ar 48000 -ac 1 <path_out>
    """
    # region command declarations
    command_with_defaults = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'afwtdn=sigma=0:levels=10:wavet=sym2:percent=85:adaptive=0:samples=8192:softness=1',
        # '-filter:a', 'afwtdn=sigma=0:levels=10:wavet=sym2:percent=85:profile=0:adaptive=0:samples=8192:softness=1',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]

    chatgpt_suggested_mild = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'afwtdn=sigma=0.005:levels=6:wavet=sym4:percent=75:adaptive=0:samples=8192:softness=1',
        # '-filter:a', 'afwtdn=sigma=0.005:levels=6:wavet=sym4:percent=75:profile=0:adaptive=0:samples=8192:softness=1',

        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]

    chatgpt_suggested_balanced = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        # '-filter:a', 'afwtdn=sigma=0.01:levels=8:wavet=coif5:percent=95:adaptive=1:samples=16384:softness=3',
        '-filter:a', 'afwtdn=sigma=0.01:levels=8:wavet=coif5:percent=95:profile=1:adaptive=0:samples=16384:softness=3',

        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]

    chatgpt_suggested_aggressive = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'afwtdn=sigma=0.02:levels=10:wavet=bl3:percent=100:adaptive=1:samples=32768:softness=5',
        # '-filter:a', 'afwtdn=sigma=0.02:levels=10:wavet=bl3:percent=100:profile=1:adaptive=1:samples=32768:softness=5',

        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]
    # endregion

    # command = command_with_defaults   # SIGSEGV
    # command = chatgpt_suggested_mild  # SIGSEGV
    command = chatgpt_suggested_balanced
    # command = chatgpt_suggested_aggressive
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, AFWTDN_PREFIX)


def ffmpeg_anlmdn(pcm_bytes, filename):
    """
    Non-Local Means algorithm. Params:
        strength, s
            Set denoising strength. Allowed range is from 0.00001 to 10000. Default value is 0.00001.
            More denoising means less noise also means less signal. I think this is a fuzzy term describing weighting/number of passes.
        patch, p
            Set patch radius duration. Allowed range is from 1 to 100 milliseconds. Default value is 2 milliseconds.
        research, r
            Set research radius duration. Allowed range is from 2 to 300 milliseconds. Default value is 6 milliseconds.
        output, o
            Set the output mode.
            It accepts the following values:
            i
                Pass input unchanged.
            o
                Pass noise filtered out.
            n
                Pass only noise.
                Default value is o.
        smooth, m
            Set smooth factor. Default value is 11. Allowed range is from 1 to 1000.

    Arthur Szlam UCLA Cam Report Recommendations for speech denoising:
        patch_size: 0.01-0.05s for speech
        search_window_size: search windows one to two times the patch size seem to work well for speech
        theta = infinity?
        number of neighbors: choose just enough neighbors so one can see this banded structure ??
    """
    strength = 0.0008
    patch_size = 0.03
    search_window = patch_size*2
    smooth_factor = 10

    command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', f'anlmdn=s={strength}:p={patch_size}:r={search_window}:m={smooth_factor}',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, ANLMDN_PREFIX)


def ffmpeg_adynamicequalizer(pcm_bytes, filename):
    """
    Technically not denoising so much as attenuating undesireable frequencies (e.g. don't belong to human speech).
    """
    mild_voice_denoising_command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'adynamicequalizer='
                     'threshold=20:'
                     'dfrequency=400:'
                     'dqfactor=2:'
                     'tfrequency=400:'
                     'tqfactor=2:'
                     'attack=20:'
                     'release=200:'
                     'ratio=1.5:'
                     'makeup=1:'
                     'range=10:'
                     'mode=1:'    # cutbelow
                     'dftype=0:'  # bandpass
                     'tftype=0:'  # bell
                     'auto=0',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]

    moderate_voice_denoising_command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'adynamicequalizer='
                     'threshold=15:'
                     'dfrequency=250:'
                     'dqfactor=3:'
                     'tfrequency=250:'
                     'tqfactor=2.5:'
                     'attack=15:'
                     'release=150:'
                     'ratio=2.5:'
                     'makeup=2:'
                     'range=20:'
                     'mode=1:'
                     'dftype=0:'
                     'tftype=0:'
                     'auto=0',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]

    aggressive_voice_denoising_command = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,       # unchanging input format
        '-ar', DESIRED_RATE,        # unchanging input rate
        '-ac', DESIRED_CHANNELS,    # unchanging input channels
        '-i', 'pipe:0',             # unchanging stdin
        '-filter:a', 'adynamicequalizer='
                     'threshold=10:'
                     'dfrequency=200:'
                     'dqfactor=4:'
                     'tfrequency=200:'
                     'tqfactor=3:'
                     'attack=10:'
                     'release=100:'
                     'ratio=4:'
                     'makeup=3:'
                     'range=30:'
                     'mode=1:'
                     'dftype=0:'
                     'tftype=0:'
                     'auto=0',
        '-f', DESIRED_FORMAT,       # unchanging output format
        'pipe:1'                    # unchaning stdout
    ]

    # command = mild_voice_denoising_command
    # command = moderate_voice_denoising_command
    command = aggressive_voice_denoising_command
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, ADYNAMICEQ_PREFIX)


def equalizer(pcm_bytes, filename):
    command = [
    'ffmpeg',
    '-f', DESIRED_FORMAT,       # unchanging input format
    '-ar', DESIRED_RATE,        # unchanging input rate
    '-ac', DESIRED_CHANNELS,    # unchanging input channels
    '-i', 'pipe:0',             # unchanging stdin
    '-filter:a', 'equalizer=f=80:t=q:w=1:g=6,equalizer=f=8000:t=q:w=1:g=5',
    '-f', DESIRED_FORMAT,       # unchanging output format
    'pipe:1'                    # unchaning stdout
    ]
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, EQUALIZER_PREFIX)

# =============================================
# endregion
# =============================================

# =============================================
# region ffmpeg audio normalization
# =============================================
def normalization_examples():
    # sample file ('aimer')
    filename = 'hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3'
    filepath = f'{ANKI_DIR}/{filename}'
    pcm_bytes = _read_audiofile_to_pcm_bytes(filepath)

    # speechnorm local normalization
    speechnorm(pcm_bytes, filename)

    # ffmpeg-normalization (e.g. loudnorm wrapper)
    external_tool_ffmpeg_normalize(pcm_bytes, filename)


def speechnorm(pcm_bytes, filename):
    """
    speechnorm works locally: it adjusts small “half-cycles” (between zero-crossings) to a peak or rms level, expanding or compressing dynamically.
    speechnorm can enhance quiet syllables or reduce harsh loud bursts — boosting clarity.
    After speechnorm, peaks might be too close to full scale (e.g., p=0.98 = –0.17 dBFS).
    ➤ Using speechnorm first smooths out micro-variability (e.g., syllables)

    loudnorm works globally: it adjusts the entire signal to a target integrated loudness, true peak, and loudness range (per ITU-R BS.1770).
    loudnorm then applies a final global adjustment — ensuring volume matches other files or playback environments (e.g., YouTube, Anki, podcasting).
    loudnorm includes true peak limiting (e.g., TP=-1.5) to catch any overshoots before final encode.
    ➤ Then loudnorm ensures overall compliance with broadcast-level targets.
    """
    command_speechnorm_default = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,  # unchanging input format
        '-ar', DESIRED_RATE,  # unchanging input rate
        '-ac', DESIRED_CHANNELS,  # unchanging input channels
        '-i', 'pipe:0',  # unchanging stdin
        '-filter:a', 'speechnorm',
        '-f', DESIRED_FORMAT,  # unchanging output format
        'pipe:1'  # unchaning stdout
    ]

    command_speechnorm_recommended = [
        'ffmpeg',
        '-f', DESIRED_FORMAT,  # unchanging input format
        '-ar', DESIRED_RATE,  # unchanging input rate
        '-ac', DESIRED_CHANNELS,  # unchanging input channels
        '-i', 'pipe:0',  # unchanging stdin
        '-filter:a', 'speechnorm=p=0.90:e=4.0:c=2.0:t=0.15:r=0.002:f=0.001:m=0.0',
        '-f', DESIRED_FORMAT,  # unchanging output format
        'pipe:1'  # unchaning stdout
    ]
    # command = command_speechnorm_default
    command = command_speechnorm_recommended
    return wrap_input_subprocess_run_with_intermediate_files(command, pcm_bytes, filename, SPEECHNORM_PREFIX)


def external_tool_ffmpeg_normalize(pcm_bytes, filename):
    """
    Normalize audio using EBU R128 loudness normalization procedure.
    """
    # convert pcm bytes to .mp3 and save intermediate files b/c ffmpeg-normalize has to work with files
    # mp3_data = _convert_pcm_to_mp3(pcm_bytes)
    filename = filename.split('.')[0] + '.wav' # pcm format
    if WRITE_INTERMEDIATE_FILES: # in this case we're technically writing either way, but if True we probably want to actually preserve
        ffmpeg_normalize_input_filepath = f'{ANKI_DIR}/{EXTERNAL_FFMPEG_NORMALIZE}_IN_{filename}' # preserve logging file if desired
        ffmpeg_normalize_output_filepath = f'{ANKI_DIR}/{EXTERNAL_FFMPEG_NORMALIZE}_OUT_{filename}' # preserve logging file if desired
    else:
        ffmpeg_normalize_input_filepath = f'/tmp/{EXTERNAL_FFMPEG_NORMALIZE}_IN_{filename}'       # oherwise tmp is fine / autodeletes on restart
        ffmpeg_normalize_output_filepath = f'/tmp/{EXTERNAL_FFMPEG_NORMALIZE}_OUT_{filename}'       # oherwise tmp is fine / autodeletes on restart

    # create input file, writes pcm bytes to .wav (ffmpeg-normalize seems to support this format best)
    command = [
        'ffmpeg',
        '-y',
        '-f', DESIRED_FORMAT,
        '-ar', DESIRED_RATE,
        '-ac', DESIRED_CHANNELS,
        '-c:a', DESIRED_CODEC,
        '-i', 'pipe:0',
        # '-f', DESIRED_FORMAT,
        # '-ar', DESIRED_RATE,
        # '-ac', DESIRED_CHANNELS,
        # '-c:a', DESIRED_CODEC,
        ffmpeg_normalize_input_filepath,
    ]
    subprocess.run(command, input=pcm_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # use external tool
    command = [
        'ffmpeg-normalize',
        ffmpeg_normalize_input_filepath,
        '-nt', 'ebu',                 # global loudness normalization
        '-t', '-16',                  # target loudness (LUFS)
        '-lrt', '7',                  # preserve some dynamic range
        '-o', ffmpeg_normalize_output_filepath,
        '--force'
    ]
    wrap_subprocess_run(command) # saves file, no output from result

    # return pcm_bytes (format will be what we wrote, no need to convert)
    pcm_bytes = _read_audiofile_to_pcm_bytes(ffmpeg_normalize_input_filepath)
    return pcm_bytes

# =============================================
# endregion
# =============================================


if __name__ == '__main__':
    main()

    # denoise_examples()
