import cv2
import os

import numpy as np

from collections import Counter
from cyolo_score_following.utils.general import load_wav, xywh2xyxy
from scipy import interpolate


SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_SIZE = 1102
FPS = SAMPLE_RATE/HOP_SIZE


def load_piece(path, piece_name):
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)

    scores = npzfile["sheets"]
    coords, systems, bars = list(npzfile["coords"]), list(npzfile['systems']), list(npzfile['bars'])

    synthesized = npzfile['synthesized'].item()
    n_pages, h, w = scores.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((0, 0), (0, 0), (pad1, pad2))

    # Add padding
    padded_scores = np.pad(scores, pad, mode="constant", constant_values=255)

    wav_path = os.path.join(path, piece_name + '.wav')
    signal = load_wav(wav_path, sr=SAMPLE_RATE)

    onsets = []
    for i in range(len(coords)):
        if coords[i]['note_x'] > 0:
            coords[i]['note_x'] += pad1

        # onset time to frame
        coords[i]['onset'] = int(coords[i]['onset'] * FPS)
        onsets.append(coords[i]['onset'])

    for i in range(len(systems)):
        systems[i]['x'] += pad1

    for i in range(len(bars)):
        bars[i]['x'] += pad1

    onsets = np.asarray(onsets, dtype=np.int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])
        bar_idx = int(Counter(onset_coords_merged['bar_idx']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        coords_new.append([note_y, note_x, system_idx, bar_idx, page_nr])
    coords_new = np.asarray(coords_new)

    # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
    # aren't played yet
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return padded_scores, scores, onsets, coords_new, bars, systems, interpol_fnc, signal, pad1, synthesized


def load_sequences(params):

    piece_idx = params['i']
    path = params['path']
    piece_name = params['piece_name']
    scale_width = params['scale_width']
    load_audio = params.get('load_audio', True)

    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = load_piece(path, piece_name)

    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    scale_factor = padded_scores.shape[1] / scale_width

    scaled_score = []
    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    scores = np.stack(scaled_score)

    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))
    piece_sequences = []

    interpol_c2o = {}
    staff_coords = {}
    add_per_staff = {}
    valid_pages = np.unique(coords_new[:, -1])

    for page_nr in valid_pages:
        page_coords = coords_new[coords_new[:, -1] == page_nr]
        page_onsets = onsets[coords_new[:, -1] == page_nr]
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords[page_nr] = sorted(np.unique(page_coords[:, 0]))

        for c in staff_coords[page_nr]:

            cs_staff = page_coords[page_coords[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff[page_nr] = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords[page_nr])):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[page_nr][idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        # if we are in between two coord position it will match it to the closest onset
        interpol_c2o[page_nr] = interpolate.interp1d(unrolled_coords_x, page_onsets, kind='nearest', bounds_error=False,
                                                     fill_value=(page_onsets[0], page_onsets[-1]))

    start_frame = 0
    curr_page = 0

    page_systems = {}
    page_bars = {}

    for page_idx in valid_pages:
        page_systems[page_idx] = list(filter(lambda x: x['page_nr'] == page_idx, systems))
        page_bars[page_idx] = list(filter(lambda bar: bar['page_nr'] == page_idx, bars))

    for frame in range(n_frames):

        true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)

        bar_idx = true_position[3]
        system_idx = true_position[2]

        # figure out at which frame we change pages
        if true_position[-1] != curr_page:
            curr_page = true_position[-1]
            start_frame = frame

        bar = bars[bar_idx]
        system = systems[system_idx]

        true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=np.float)
        true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=np.float)

        systems_xywh = np.asarray([[x['x'], x['y'], x['w'], x['h']] for x in page_systems[curr_page]])
        systems_xyxy = xywh2xyxy(systems_xywh)

        max_x_shift = (-(int(systems_xyxy[:, 0].min() - 50)),
                       int(padded_scores.shape[2] - systems_xyxy[:, 2].max() - 50))
        max_y_shift = (min(0, -int((systems_xyxy[:, 1].min() - 50))),
                       max(1, int(padded_scores.shape[1] - systems_xyxy[:, 3].max() - 50)))

        piece_sequences.append({'piece_id': piece_idx,
                                'is_onset': frame in onsets,
                                'start_frame': start_frame,
                                'frame': frame,
                                'max_x_shift': max_x_shift,
                                'max_y_shift': max_y_shift,
                                'true_position': true_position,
                                'true_system': true_system,
                                'true_bar': true_bar,
                                'height': system['h'],
                                'synthesized': synthesized,
                                'scale_factor': scale_factor,
                                })

    if not load_audio:
        signal = os.path.join(path, piece_name + '.wav')

    return piece_idx, scores, signal, piece_name, piece_sequences, interpol_c2o, staff_coords, add_per_staff


def load_piece_for_testing(path, piece_name, scale_width):

    padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _ = load_piece(path, piece_name)

    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    # scale scores
    scaled_score = []
    scale_factor = scores[0].shape[0] / scale_width

    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    score = np.stack(scaled_score)

    org_scores_rgb = []
    for org_score in org_scores:
        org_score = np.array(org_score, dtype=np.float32) / 255.

        org_scores_rgb.append(cv2.cvtColor(org_score, cv2.COLOR_GRAY2BGR))

    return org_scores_rgb, score, signal, systems, bars, interpol_fnc, pad, scale_factor, onsets
