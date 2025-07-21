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
import io
from base64 import encode

# neural nine imports
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.fftpack as fft
from scipy.signal import medfilt


# ==== Configuration ====
BACKUP = False                      # todo toggle to True prior to pushing
WRITE_INTERMEDIATE_FILES = False    # True is helpful for fine tuning what each function dones
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

# File Naming Constants
USER_PATH = os.path.expanduser('~')
ANKI_DIR = f'{USER_PATH}/.local/share/Anki2/User 1/collection.media'
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
    if BACKUP:
        backup_audio_collection()

    filenames = ['hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3']
    for filename in filenames:
        # read audio file into required format
        filepath = f'{ANKI_DIR}/{filename}'
        pcm_bytes = _read_mp3_to_pcm_bytes(filepath)

        # clean up audio using ffmpeg wrappers
        audio_chain(pcm_bytes, filename)

        # write audio to mp3 file
        mp3_data = _convert_pcm_to_mp3(pcm_bytes)
        filepath = f'{ANKI_DIR}/SUCCESS_{filename}'
        with open(filepath, 'wb') as f:
            f.write(mp3_data)


def audio_chain(pcm_bytes, filename):
    # 1. remove silence
    pcm_bytes = ffmpeg_silenceremove(pcm_bytes, filename)

    # 2. remove non-speech (coughing, sniffling, shuffling, humming, etc) - note: model works best early in process
    sr_model_path = 'resources/rnnoise_models/speech_recording.rnnn'
    pcm_bytes = ffmpeg_arnndn(pcm_bytes, filename, sr_model_path)

    # 3. high/low pass filter helps with extreme noises
    pcm_bytes = ffmpeg_lowpass_highpass(pcm_bytes, filename)

    # 4. x3 rounds of afftdn for general clarity
    pcm_bytes = ffmpeg_afftdn(pcm_bytes, filename)
    pcm_bytes = ffmpeg_afftdn(pcm_bytes, filename)
    pcm_bytes = ffmpeg_afftdn(pcm_bytes, filename)

    # 5. stft is good at cleaning front and tail
    pcm_bytes = neural_nine_demo(pcm_bytes, filename)

    # skip local volume normalization - it ruins emphasis

    # 6. global volume normalization
    step6_bytes = external_tool_ffmpeg_normalize(pcm_bytes, filename)

    # 7. rebrighten a bit
    pcm_bytes = equalizer(step6_bytes, filename)
    return pcm_bytes


def backup_audio_collection():
    t = datetime.datetime.now()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    src_collection = f'{USER_PATH}/.local/share/Anki2/User 1/collection.media'
    dst_collection = f'{USER_PATH}/.local/share/Anki2/User 1/collection.media.backup_{timestamp}'
    shutil.copytree(src_collection, dst_collection, dirs_exist_ok=True) # overwriting is a-okay :)


def run_command(command, pcm_bytes, filename, prefix=None):
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
        print('FFmpeg stderr:\n')
        print(e.stderr.decode('utf-8'))
        raise

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
    return run_command(command, pcm_bytes, filename, SILENCE_RM_PREFIX)

# =============================================
# region ffmpeg audio filters
# =============================================
def denoise_examples():
    filename = 'hypertts-6840f600086fd031f601da7d58c4e6ab98b511d2c9242193b5ecc55b.mp3' # sample is: 'aimer'
    filepath = f'{ANKI_DIR}/{filename}'

    # region short time fourier transform filtering from youtuber 'Neural Nine'
    n9_wbq_out = f'{ANKI_DIR}/{NEURAL_NINE_PREFIX}_{filename}'
    neural_nine_demo(filepath, n9_wbq_out)
    # endregion

    # region neural network rnnoise models
    model_path = 'resources/rnnoise_models'
    gg_arnndn_wbq_out = f'{ANKI_DIR}/{ARNNDN_PREFIX}_gg_{filename}'
    gg_model = f'{model_path}/general_general.rnnn'
    gr_arnndn_wbq_out = f'{ANKI_DIR}/{ARNNDN_PREFIX}_gr_{filename}'
    gr_model = f'{model_path}/general_recording.rnnn'
    vg_arnndn_wbq_out = f'{ANKI_DIR}/{ARNNDN_PREFIX}_vg_{filename}'
    vg_model = f'{model_path}/voice_general.rnnn'
    vr_arnndn_wbq_out = f'{ANKI_DIR}/{ARNNDN_PREFIX}_vr_{filename}'
    vr_model = f'{model_path}/voice_recording.rnnn'
    sr_arnndn_wbq_out = f'{ANKI_DIR}/{ARNNDN_PREFIX}_sr_{filename}'
    sr_model = f'{model_path}/speech_recording.rnnn'

    ffmpeg_arnndn(filepath, gg_arnndn_wbq_out, gg_model)
    ffmpeg_arnndn(filepath, gr_arnndn_wbq_out, gr_model)
    ffmpeg_arnndn(filepath, vg_arnndn_wbq_out, vg_model)
    ffmpeg_arnndn(filepath, vr_arnndn_wbq_out, vr_model)
    ffmpeg_arnndn(filepath, sr_arnndn_wbq_out, sr_model)
    # endregion

    # region S.O. two afftdn passes followed by highpass & lowpass filter
    lowpass_highpass_out = f'{ANKI_DIR}/{LHPASS_PREFIX}_{filename}'
    afftdn_out1 = f'{ANKI_DIR}/{AFFTDN_PREFIX}_pass1_{filename}'
    afftdn_out2 = f'{ANKI_DIR}/{AFFTDN_PREFIX}_pass2_{filename}'
    ffmpeg_lowpass_highpass(filepath, lowpass_highpass_out)
    ffmpeg_afftdn(lowpass_highpass_out, afftdn_out1)
    ffmpeg_afftdn(afftdn_out1, afftdn_out2)
    # endregion

    # adeclick - impulsive filtering
    adeclick_out = f'{ANKI_DIR}/{ADECLICK_PREFIX}_{filename}'
    ffmpeg_adeclick(filepath, adeclick_out)

    # afwtdn - broadband filtering
    # I found this one to be uniquely bad. Maybe I just didn't tweak it enough.
    afwtdn_out = f'{ANKI_DIR}/{AFWTDN_PREFIX}_{filename}'
    ffmpeg_afwtdn(filepath, afwtdn_out)

    # anldmn - broadband filtering (ref UCLA math article/heat equation)
    anlmdn_out = f'{ANKI_DIR}/{ANLMDN_PREFIX}_{filename}'
    ffmpeg_anlmdn(filepath, anlmdn_out)

    # adynamicequalizer - attenuate unwanted freqs
    adynamicequalizer_out = f'{ANKI_DIR}/{ADYNAMICEQ_PREFIX}_{filename}'
    ffmpeg_adynamicequalizer(filepath, adynamicequalizer_out)

    return


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
    mask = medfilt(mask, kernel_size=(1,5))
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
    result = subprocess.run(command, input=buffer_pcm_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout # pcm bytes in exactly our required format


def ffmpeg_arnndn(pcm_bytes, filename, rel_model_path, mix=1):
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
    return run_command(command, pcm_bytes, filename, f'{ARNNDN_PREFIX}_sr_')


def _read_mp3_to_pcm_bytes(input_path) -> bytes:
    command = [
        'ffmpeg',
        '-i', input_path,
        '-f', DESIRED_FORMAT,
        '-acodec', DESIRED_CODEC,
        '-ac', DESIRED_CHANNELS,
        '-ar', DESIRED_RATE,
        'pipe:1'  # stdout
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout  # raw PCM bytes


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
    return run_command(command, pcm_bytes, filename, AFFTDN_PREFIX)


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
    return run_command(command, pcm_bytes, filename, LHPASS_PREFIX)


def ffmpeg_adeclick(path_in, path_out):
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
        '-y',
        '-i', path_in,
        '-filter:a', 'adeclick=w=10:o=95:a=0:t=1:b=0:m=s',
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
        '-y',
        '-i', path_in,
        '-filter:a', f'anlmdn=s={strength}:p={patch_size}:r={search_window}:m={smooth_factor}',
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
    return run_command(command, pcm_bytes, filename, EQUALIZER_PREFIX)

# =============================================
# endregion
# =============================================

# =============================================
# region ffmpeg audio normalization
# =============================================
def normalization_examples():
    # speechnorm local normalization
    speechnorm_out = f'{ANKI_DIR}/{SPEECHNORM_PREFIX}_{woman_broadband_quiet}'
    speechnorm(wbq_in_path, speechnorm_out)

    # ffmpeg-normalization (e.g. loudnorm wrapper)
    final_out = f'{ANKI_DIR}/final_{woman_broadband_quiet}'
    external_tool_ffmpeg_normalize(wbq_in_path, final_out)


def speechnorm(path_in, path_out):
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
        '-y',
        '-i', path_in,
        '-filter:a', 'speechnorm',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    command_speechnorm_recommended = [
        'ffmpeg',
        '-y',
        '-i', path_in,
        '-filter:a', 'speechnorm=p=0.90:e=4.0:c=2.0:t=0.15:r=0.002:f=0.001:m=0.0',
        '-acodec', 'libmp3lame',
        '-ar', '48000',
        '-ac', '1',
        path_out
    ]

    # command = command_speechnorm_default
    command = command_speechnorm_recommended
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise


def external_tool_ffmpeg_normalize(pcm_bytes, filename):
    """
    Normalize audio using EBU R128 loudness normalization procedure.
    """

    # convert pcm bytes to .mp3 and save intermediate files b/c ffmpeg-normalize has to work with files
    mp3_data = _convert_pcm_to_mp3(pcm_bytes)
    if WRITE_INTERMEDIATE_FILES: # in this case we're technically writing either way, but if True we probably want to actually preserve
        filepath = f'{ANKI_DIR}/{EXTERNAL_FFMPEG_NORMALIZE}_{filename}.mp3' # preserve logging file if desired
    else:
        filepath = f'/tmp/{EXTERNAL_FFMPEG_NORMALIZE}_{filename}.mp3'       # oherwise tmp is fine / autodeletes on restart
    with open(filepath, 'wb') as f:
        f.write(mp3_data)

    # use external tool
    path_in = filepath
    path_out = filepath
    command = [
        'ffmpeg-normalize',
        path_in,
        # '-f', WRITE_FORMAT,         # input format
        '-c:a', WRITE_CODEC,
        '-ar', WRITE_RATE,          # input rate
        '-ac', WRITE_CHANNELS,      # input channels
        '-o', path_out,
        '-nt', 'ebu',               # global loudness normalization
        '-t', '-23',                # target loudness (LUFS)
        '-lrt', '7',                # preserve some dynamic range
        # '-acodec', DESIRED_CODEC,   # output codec
        # '-q:a', '3',              # todo VBR quality (2=~190kbps, 3=~175kbps) - this arg breaks ffmpeg-normalize for some reason, leave default
        '-ar', DESIRED_RATE,        # output sample rate
        '-ac', DESIRED_CHANNELS,    # output channels (mono output)
        '--force'
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg stderr:\n")
        print(e.stderr.decode('utf-8'))
        raise

    # return pcm_bytes (format will be what we wrote, no need to convert)
    pcm_bytes = _read_mp3_to_pcm_bytes(filepath)
    return pcm_bytes

# =============================================
# endregion
# =============================================


if __name__ == '__main__':
    main()

    # denoise_examples()
