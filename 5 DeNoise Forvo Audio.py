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
BACKUP = False # todo toggle to True prior to pushing
# ========================

USER_PATH = os.path.expanduser('~')
ANKI_DIR = f'{USER_PATH}/.local/share/Anki2/User 1/collection.media'
NEURAL_NINE_PREFIX = 'n9stft_'
ARNNDN_PREFIX = 'arnndn_'
AFFTDN_PREFIX = 'afftdn_'
LHPASS_PREFIX = 'lhpass_'
ADECLICK_PREFIX = 'adeclick_'
AFWTDN_PREFIX = 'afwtdn_broadband_'
ANLMDN_PREFIX = 'anlmdn_broadband_least_'
ADYNAMICEQ_PREFIX = 'adynamiceq_'


# aimer
woman_broadband_quiet = 'hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3'
wbq_in_path = f'{ANKI_DIR}/{woman_broadband_quiet}'

'''
hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3
'''

def main():
    if BACKUP:
        backup_audio_collection()

    # potential chains
    # region x2 afftdn, 1 lp/hp - good at... everything
    ffmpeg_afftdn(wbq_in_path, afftdn_path_out1)
    ffmpeg_afftdn(afftdn_path_out1, afftdn_path_out2)
    ffmpeg_lowpass_highpass(afftdn_path_out2, lowpass_highpass_path_out3)
    # endregion

    ffmpeg_arnndn(wbq_in_path, sr_arnndn_wbq_out_path, sr_model) # good at noise removal throughout
    neural_nine_demo() # good at cleaning front and tail

    return


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
    ffmpeg_lowpass_highpass(afftdn_path_out2, lowpass_highpass_path_out3)
    # endregion

    # adeclick - impulsive filtering
    adeclick_out = f'{ANKI_DIR}/{ADECLICK_PREFIX}_{woman_broadband_quiet}'
    ffmpeg_adeclick(wbq_in_path, adeclick_out)

    # afwtdn - broadband filtering
    # I found this one to be uniquely bad. Maybe I just didn't tweak it enough.
    afwtdn_out = f'{ANKI_DIR}/{AFWTDN_PREFIX}_{woman_broadband_quiet}'
    ffmpeg_afwtdn(wbq_in_path, afwtdn_out)

    # anldmn - broadband filtering (ref UCLA math article/heat equation)
    anlmdn_out = f'{ANKI_DIR}/{ANLMDN_PREFIX}_{woman_broadband_quiet}'
    ffmpeg_anlmdn(wbq_in_path, anlmdn_out)

    # adynamicequalizer - attenuate unwanted freqs
    adynamicequalizer_out = f'{ANKI_DIR}/{ADYNAMICEQ_PREFIX}_{woman_broadband_quiet}'
    ffmpeg_adynamicequalizer(wbq_in_path, adynamicequalizer_out)

    return


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
        '-y',
        '-i', path_in,
        '-filter:a', 'afftdn=nf=-25',
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
        '-y',
        '-i', path_in,
        '-filter:a', f'highpass=f=200, lowpass=f=3000',
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


def ffmpeg_adeclick(path_in, path_out):
    """
    reference command:
    ffmpeg -y -i <path_in> -af 'adeclick=w=60:o=85:a=10:t=3:b=5:m=s'
     -acodec libmp3lame -ar 48000 -ac 1 <path_out>

    Filter explanation (gist: uses slightly more aggressive noise removal)
    -af 'adeclick=w=60:o=85:a=10:t=3:b=5:m=s'
         │   │   │   │   │   └── overlap-save method
         │   │   │   │   └───── burst fusion = 5% of window
         │   │   │   └───────── threshold = 3
         │   │   └───────────── autoregression order = 10% of window
         │   └───────────────── overlap = 85%
         └───────────────────── window size = 60 ms
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', path_in,
        '-filter:a', 'adeclick=w=60:o=85:a=10:t=3:b=5:m=s',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


def ffmpeg_afwtdn(path_in, path_out):
    """
    I hate this one. Maybe tweaking would help? Maybe not.

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
        '-y',
        '-i', path_in,
        '-filter:a', 'afwtdn=sigma=0:levels=10:wavet=sym2:percent=85:adaptive=0:samples=8192:softness=1',
        # '-filter:a', 'afwtdn=sigma=0:levels=10:wavet=sym2:percent=85:profile=0:adaptive=0:samples=8192:softness=1',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    chatgpt_suggested_mild = [
        'ffmpeg',
        '-y',
        '-i', path_in,
        '-filter:a', 'afwtdn=sigma=0.005:levels=6:wavet=sym4:percent=75:adaptive=0:samples=8192:softness=1',
        # '-filter:a', 'afwtdn=sigma=0.005:levels=6:wavet=sym4:percent=75:profile=0:adaptive=0:samples=8192:softness=1',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    chatgpt_suggested_balanced = [
        'ffmpeg',
        '-y',
        '-i', path_in,
        # '-filter:a', 'afwtdn=sigma=0.01:levels=8:wavet=coif5:percent=95:adaptive=1:samples=16384:softness=3',
        '-filter:a', 'afwtdn=sigma=0.01:levels=8:wavet=coif5:percent=95:profile=1:adaptive=0:samples=16384:softness=3',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    chatgpt_suggested_aggressive = [
        'ffmpeg',
        '-y',
        '-i', path_in,
        '-filter:a', 'afwtdn=sigma=0.02:levels=10:wavet=bl3:percent=100:adaptive=1:samples=32768:softness=5',
        # '-filter:a', 'afwtdn=sigma=0.02:levels=10:wavet=bl3:percent=100:profile=1:adaptive=1:samples=32768:softness=5',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]
    # endregion

    # command = command_with_defaults   # SIGSEGV
    # command = chatgpt_suggested_mild  # SIGSEGV
    command = chatgpt_suggested_balanced
    # command = chatgpt_suggested_aggressive
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


def ffmpeg_anlmdn(path_in, path_out):
    """
    todo
        The expected values don't match the documentation for both p= and r=. Wack.
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', path_in,
        '-filter:a', 'anlmdn=s=0.00001:p=0.01:r=0.002:o=o:m=11',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


def ffmpeg_adynamicequalizer(path_in, path_out):
    """
    Technically not denoising so much as attenuating undesireable frequencies (e.g. don't belong to human speech).
    """
    mild_voice_denoising_command = [
        'ffmpeg', '-y',
        '-i', path_in,
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
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    moderate_voice_denoising_command = [
        'ffmpeg', '-y',
        '-i', path_in,
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
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    aggressive_voice_denoising_command = [
        'ffmpeg', '-y',
        '-i', path_in,
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
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    # command = mild_voice_denoising_command
    # command = moderate_voice_denoising_command
    command = aggressive_voice_denoising_command
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


if __name__ == '__main__':
    main()