import librosa
import os
import random

from cyolo_score_following.utils.data_utils import SAMPLE_RATE
from multiprocessing import get_context
from pathlib import Path
from scipy.signal import convolve


# filter some impulse responses that produce a weird distorted sound
FILTER_LIST = [
    "1a_marble_hall.wav",
    "3a_hats_cloaks_the_lord.wav",
    "4a_hats_cloaks_visitors.wav",
    "18a_smoking_room.wav",
    "ir_-_location_1_s1_-_r1.wav",
    "ir_-_location_2_s1_-_r2.wav",
    "ir_-_location_3_s1_-_r3.wav",
    "ir_-_location_4_s2_-_r3.wav",
    "ir_-_location_5_s2_-_r1.wav",
    "ir_-_location_6_s2_-_r4.wav",
    "ir_p1.wav",
    "ir_p2.wav",
    "ir_p3.wav",
    "ir_p4.wav",
    "ir_p5.wav",
    "ir_p6.wav",
    "ir_posic2-pb.wav",
    "ir_posic2-pb_-_sfdc.wav",
    "ir_posic3-1erpiso_-_sfdc.wav",
    "ir_posic3-pb_-_sfdc.wav",
    "ir_posic4-1erpiso_-_sfdc.wav",
    "ir_posic5-1erpiso_-_sfdc.wav",
    "ir_posic5-pb_-_sfdc.wav",
    "ir_stage_-_sfdc.wav",
    "phase1_bformat.wav",
    "phase1_bformat_catt.wav",
    "phase1_stereo.wav",
    "phase2_bformat.wav",
    "phase2_stereo.wav",
    "phase3_bformat.wav",
    "phase3_stereo.wav",
    "slinky_ir.wav",
    "sp1_mp4_ir_stereo_trimmed.wav",
    "sp2_mp2_ir_stereo_trimmed.wav",
    "sp2_mp4_ir_bformat_trimmed.wav",
    "sp2_mp4_ir_stereo_trimmed.wav",
    "spokane_womans_club_ir.wav",
    "stairwell_ir_bformat_trimmed.wav",
    "stairwell_ir_stereo_trimmed.wav",
    "tsfthr.wav"
]


def load_signal(path):
    signal, _ = librosa.load(path, sr=SAMPLE_RATE)

    return signal


def load_irs(ir_paths):
    print("Loading IRs from", ir_paths)
    all_paths = []
    for ir_path in ir_paths:
        all_paths.extend([path for path in Path(os.path.expanduser(ir_path)).rglob('*.wav')])

    # filter virtual-membranes
    all_paths = list(filter(lambda x: "virtual-membranes" not in str(x), all_paths))

    arguments = []
    for path in all_paths:

        if os.path.basename(path._str) not in FILTER_LIST:
            arguments.append(path._str)

    with get_context("fork").Pool(8) as pool:
        irs = pool.map(load_signal, arguments)

    return irs, all_paths


class ImpulseResponse(object):

    def __init__(self, ir_paths, ir_prob):
        self.ir_prob = ir_prob

        self.irs, self.ir_paths = load_irs(ir_paths)

        print(f'Loaded {len(self.irs)} impulse responses.')

    def __call__(self, sample):

        performance = sample['performance']

        if random.random() < self.ir_prob:
            ir = random.choices(self.irs, k=1)[0]
            performance = convolve(performance, ir, 'full')[:-(ir.shape[0] - 1)]

            sample['performance'] = performance

        return sample
