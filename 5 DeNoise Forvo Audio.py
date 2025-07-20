"""
Provide a variety of free denoising options.

I'm just going to the leave the best options in main() which will of course write the new files.

Warning: This will overwrite the old files sound files. However, by default all files will be
           backed up in collection.media.backup_{timestamp}. This may lead to a rather large
           amount of duplicate data. If you prefer to disable this option toggle the configuration
           BACKUP=True to BACKUP=False.
"""
import os
import subprocess
import shutil
import datetime
import time

# neural nine imports
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.fftpack as fft
from scipy.signal import medfilt


# ==== Configuration ====
BACKUP = True # todo toggle to True prior to pushing
# ========================

USER_PATH = os.path.expanduser('~')
ANKI_DIR = f'{USER_PATH}/.local/share/Anki2/User 1/collection.media'
NEURAL_NINE_PREFIX = 'n9_'
ARNNDN_PREFIX = 'arnndn_'
AFFTDN_PREFIX = 'afftdn_'
LHPASS_PREFIX = 'lhpass_'

# aimer
woman_broadband_quiet = 'hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3'
wbq_in_path = f'{ANKI_DIR}/{woman_broadband_quiet}'

'''
hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3
'''

def main():
    if BACKUP:
        backup_audio_collection()


    examples()


def examples():
    # region short time fourier transform filtering from youtuber 'Neural Nine'
    n9_wbq_out_path = f'{ANKI_DIR}/{NEURAL_NINE_PREFIX}_{woman_broadband_quiet}'
    neural_nine_demo(wbq_in_path, n9_wbq_out_path)
    # endregion

    # region neural network rnnoise models
    model_path = 'resources/rnnoise_models'
    gg_arnndn_wbq_out_path = f'{ANKI_DIR}/{ARNNDN_PREFIX}_gg_{woman_broadband_quiet}'
    gg_model = f'{model_path}/general_general.rnnn'
    gr_arnndn_wbq_out_path = f'{ANKI_DIR}/{ARNNDN_PREFIX}_gr_{woman_broadband_quiet}'
    gr_model = f'{model_path}/general_recording.rnnn'
    vg_arnndn_wbq_out_path = f'{ANKI_DIR}/{ARNNDN_PREFIX}_vg_{woman_broadband_quiet}'
    vg_model = f'{model_path}/voice_general.rnnn'
    vr_arnndn_wbq_out_path = f'{ANKI_DIR}/{ARNNDN_PREFIX}_vr_{woman_broadband_quiet}'
    vr_model = f'{model_path}/voice_recording.rnnn'
    sr_arnndn_wbq_out_path = f'{ANKI_DIR}/{ARNNDN_PREFIX}_sr_{woman_broadband_quiet}'
    sr_model = f'{model_path}/speech_recording.rnnn'

    ffmpeg_arnndn(wbq_in_path, gg_arnndn_wbq_out_path, gg_model)
    ffmpeg_arnndn(wbq_in_path, gr_arnndn_wbq_out_path, gr_model)
    ffmpeg_arnndn(wbq_in_path, vg_arnndn_wbq_out_path, vg_model)
    ffmpeg_arnndn(wbq_in_path, vr_arnndn_wbq_out_path, vr_model)
    ffmpeg_arnndn(wbq_in_path, sr_arnndn_wbq_out_path, sr_model)
    # endregion

    # region S.O. two afftdn passes followed by highpass & lowpass filter
    afftdn_path_out1 = f'{ANKI_DIR}/{AFFTDN_PREFIX}_pass1_{woman_broadband_quiet}'
    afftdn_path_out2 = f'{ANKI_DIR}/{AFFTDN_PREFIX}_pass2_{woman_broadband_quiet}'
    lowpass_highpass_path_out3 = f'{ANKI_DIR}/{LHPASS_PREFIX}_{woman_broadband_quiet}'
    ffmpeg_afftdn(wbq_in_path, afftdn_path_out1)
    ffmpeg_afftdn(afftdn_path_out1, afftdn_path_out2)
    # todo hanging at ffmpeg_lowpass_highpass(afftdn_path_out2, lowpass_highpass_path_out3)
    # endregion

    # adeclick - impulsive filtering

    # afwtdn / anldmn - broadband filtering

    #

    #


def backup_audio_collection():
    t = datetime.datetime.now()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    src_collection = f'{USER_PATH}/.local/share/Anki2/User 1/collection.media'
    dst_collection = f'{USER_PATH}/.local/share/Anki2/User 1/collection.media.backup_{timestamp}'
    shutil.copytree(src_collection, dst_collection, dirs_exist_ok=True) # overwriting is a-okay :)


def neural_nine_demo(path_in, path_out):
    """
    FFT denoising written by NeuralNine (youtuber).
    Seems to be geared toward broadband filtering rather than impulse.

    Results... not loving it.
    """
    y, sr = librosa.load(path_in, sr=None)
    S_full, phase = librosa.magphase(librosa.stft(y))
    noise_power = np.mean(S_full[:, :int(sr*0.1)], axis=1)
    mask = S_full > noise_power[:, None]
    mask = mask.astype(float)
    mask = medfilt(mask, kernel_size=(1,5))
    S_clean = S_full * mask
    y_clean = librosa.istft(S_clean * phase)
    sf.write(path_out, y_clean, sr)
    return


def ffmpeg_arnndn(path_in, path_out, rel_model_path, mix=1):
    """
    arnndn params:
        model   : the model file (.rnnn)
        mix     : 0 is original, 1 is filtered, (0-1) is a blend of original and filtered, negative values are what was filtered out rather than filtered for (e.g. -1 is just noise)

    Equivalent cli command:
        ffmpeg -i <input_file.mp3> -filter:a arnndn=model=<model_path.rnnn>:mix
    """
    # fix model not found in subprocess calls
    abs_model_path = os.path.abspath(rel_model_path)

    # rnnoise operates on RAW 16-bit (machine endian) mono - convert mp3
    raw_pcm_bytes = _convert_mp3_to_pcm_bytes(path_in)
    denoise_pcm_to_mp3(raw_pcm_bytes, path_out, abs_model_path, mix)


def _convert_mp3_to_pcm_bytes(input_path: str) -> bytes:
    command = [
        'ffmpeg',
        '-i', input_path,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', '48000',
        'pipe:1'  # stdout
    ]

    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout  # Raw PCM bytes


def denoise_pcm_to_mp3(pcm_bytes: bytes, path_out, model_path, mix=1):
    command = [
        'ffmpeg',
        '-f', 's16le',             # raw PCM format
        '-ar', '48000',            # sample rate
        '-ac', '1',                # mono
        '-i', 'pipe:0',            # input from stdin
        '-filter:a', f'arnndn=model={model_path}:mix={mix}',
        '-acodec', 'libmp3lame',   # MP3 encoder
        '-f', 'mp3',               # output format
        path_out                   # write file. to write to stdout replace line with 'pipe:1'
    ]

    try:
        result = subprocess.run(
            command,
            input=pcm_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


def ffmpeg_afftdn(path_in, path_out):
    """
    Sample frequencies referenced from Matteo M. on S.O.

    reference command:
    ffmpeg -i <path_in> -af "afftdn=nf=-25" <path_out>
    """
    command = [
        'ffmpeg',
        '-i', path_in,
        '-filter:a', f'afftdn=nf=-25',
        '-acodec', 'libmp3lame',        # MP3 encoder
        '-f', 'mp3',                    # output format
        '-ac', '1',
        '-ar', '48000',
        path_out
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


def ffmpeg_lowpass_highpass(path_in, path_out):
    """
    Sample frequencies referenced from Matteo M. on S.O.

    reference command:
    ffmpeg -i <path_in> -af "lowpass=f=3000, highpass=f=200" <path_out>
    """
    command = [
        'ffmpeg',
        '-i', path_in,
        '-filter:a', f'highpass=f=200, lowpass=f=300',
        '-acodec', 'libmp3lame',        # MP3 encoder
        '-f', 'mp3',                    # output format
        '-ac', '1',
        '-ar', '48000',
        path_out
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


def adeclick():
    return

def afwtdn():
    return

def anlmdn():
    return

def libvmaf():
    return


if __name__ == '__main__':
    main()