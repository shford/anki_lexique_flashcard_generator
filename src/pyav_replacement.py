import io
import json
import os
import subprocess
import time

import av

from s8_DeNoise_Forvo_Audio import SELECTED_AUDIO_PREFIX
from s8_DeNoise_Forvo_Audio import ALT_TXT_IN_AUDIO
from s8_DeNoise_Forvo_Audio import WRITE_FORMAT
from s8_DeNoise_Forvo_Audio import USER_PATH
from s8_DeNoise_Forvo_Audio import PROFILE


PYAV_TEST_DIR = f'{USER_PATH}/.local/share/Anki2/{PROFILE}/PYAV_TEST_DIR'


def main():
    t1 = time.time()

    dir_contents = os.listdir(PYAV_TEST_DIR)
    hypertts_mp3_filenames = [f for f in dir_contents if (os.path.isfile(os.path.join(PYAV_TEST_DIR, f)) and SELECTED_AUDIO_PREFIX in f)]
    std_forvo_api_mp3_filenames = [f for f in dir_contents if (os.path.isfile(os.path.join(PYAV_TEST_DIR, f)) and ALT_TXT_IN_AUDIO in f and WRITE_FORMAT in f and not 'ATTS ' in f)]
    mp3_filenames = hypertts_mp3_filenames + std_forvo_api_mp3_filenames

    print(f'Executing denoising program on {len(mp3_filenames)} files.\n')
    # with Pool(processes=cpu_count()) as pool:
    #     pool.map(process_audio_file, mp3_filenames)
    for filename in mp3_filenames:
        process_file(filename)

    t2 = time.time()
    print(f'\nWrote {len(mp3_filenames)} new files in {t2-t1} seconds.\n.')


def measure_loudness_pcm(pcm_bytes, channels, rate, target_lufs=-16, lra=7, tp=-1.5):
    '''measure EBU R128 loudness on raw PCM bytes'''
    cmd = [
        'ffmpeg', '-hide_banner', '-nostdin',
        '-f', 's16le', '-ar', str(rate), '-ac', str(channels),
        '-i', 'pipe:0',
        '-af', f'loudnorm=I={target_lufs}:LRA={lra}:TP={tp}:print_format=json',
        '-f', 'null', '-'
    ]
    p = subprocess.run(cmd, input=pcm_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    stderr = p.stderr.decode('utf-8', errors='ignore')
    i = stderr.find('{')
    j = stderr.rfind('}')
    if i == -1 or j == -1:
        raise RuntimeError('could not parse loudnorm json from ffmpeg stderr')
    return json.loads(stderr[i:j+1])


def process_file(filename):
    # 1) decode input mp3 to raw PCM frames
    input_path = f'{PYAV_TEST_DIR}/{filename}'
    with av.open(input_path) as container:
        input_stream = container.streams.audio[0]
        sr = input_stream.codec_context.sample_rate
        ch = input_stream.codec_context.channels
        pcm_frames = [frame for frame in container.decode(audio=0)]

    # 2) first graph: arnndn -> adeclick
    graph_one = av.filter.Graph()
    rnnn_model_path = '../resources/rnnoise_models/speech_recording.rnnn'

    graph_one.link_nodes(
        graph_one.add_abuffer(template=input_stream),
        add_arnndn_to_graph(graph_one, rnnn_model_path),
        add_adeclick_to_graph(graph_one),
        graph_one.add('abuffersink'),
    ).config()

    # push frames through graph 1, collect raw PCM in memory
    pcm_bytes = bytearray()
    for frame in pcm_frames:
        graph_one.push(frame)
        while True:
            f = graph_one.pull()
            if f is None:
                break
            pcm_bytes.extend(f.planes[0].to_bytes())

    # 3) first-pass loudnorm measurement
    meas = measure_loudness_pcm(pcm_bytes, channels=ch, rate=sr, target_lufs=LUFS_NORMALIZATION_LEVEL)

    # 4) second graph: loudnorm -> equalizer -> mp3
    with av.open(output_path, mode='w', format='mp3') as out_mp3:
        mp3_stream = out_mp3.add_stream('libmp3lame', rate=sr, channels=ch)

        graph2 = av.filter.Graph()
        src2 = graph2.add_buffer(template=input_stream)
        sink2 = graph2.add('abuffersink')

        loudnorm_node = add_loudnorm_to_graph(graph2, )
        equalizer_node = add_equalizer_to_graph(graph2)

        # link second graph
        src2.link_to(loudnorm_node)
        graph2.link_to(equalizer_node)
        graph2.link_to(sink2)
        graph2.configure()

        # feed the original PCM frames into the second graph
        for frame in pcm_frames:
            graph2.push(frame)
            while True:
                f = graph2.pull()
                if f is None:
                    break
                for pkt in mp3_stream.encode(f):
                    out_mp3.mux(pkt)
        for pkt in mp3_stream.encode():
            out_mp3.mux(pkt)


# -------------------------
# filter graph helpers
# -------------------------
def add_arnndn_to_graph(graph, rel_model_path, mix=1.0):
    abs_model_path = os.path.abspath(rel_model_path)
    return graph.add('arnndn', args=f'model={abs_model_path}:mix={mix}')

def add_adeclick_to_graph(graph):
    adeclick_args = (
        'adeclick=w=20:o=90:a=3:t=1.0:b=3:m=a'
    )
    return graph.add('adeclick', args=adeclick_args)

def add_equalizer_to_graph(graph):
    bands = [
        dict(f=120,   w=2,   g=3),
        dict(f=800,   w=1.5, g=2),
        dict(f=3500,  w=2,   g=4),
        dict(f=6000,  w=2,   g=3),
        dict(f=11000, w=1.5, g=2),
    ]
    node = None
    for b in bands:
        eq_node = graph.add('equalizer', args=f'f={b["f"]}:t=q:w={b["w"]}:g={b["g"]}')
        if node is not None:
            node.link_to(eq_node)
        node = eq_node
    return node

def add_loudnorm_to_graph(graph, ):
    ln_args = (
        f'I={LUFS_NORMALIZATION_LEVEL}:LRA=7:TP=-1.5:'
        f'measured_I={meas["input_i"]}:measured_LRA={meas["input_lra"]}:'
        f'measured_TP={meas["input_tp"]}:measured_thresh={meas["input_thresh"]}:'
        f'offset={meas["target_offset"]}:linear=true:print_format=summary'
    )
    ln_node = graph.add('loudnorm', args=ln_args)


if __name__ == '__main__':
    main()