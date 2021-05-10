
import glob
import os
import random
import torch
import torchvision

import numpy as np


from cyolo_score_following.utils.data_utils import load_sequences, SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from cyolo_score_following.utils.dist_utils import is_main_process
from cyolo_score_following.utils.general import load_wav, AverageMeter, get_max_box
from cyolo_score_following.augmentations.impulse_response import ImpulseResponse
from cyolo_score_following.utils.general import load_yaml

from multiprocessing import get_context
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class SequenceDataset(Dataset):
    def __init__(self, scores, performances, sequences, piece_names, interpol_c2o,
                 staff_coords, add_per_staff, augment=False, transform=None):

        self.scores = scores
        self.performances = performances
        self.sequences = []
        self.rand_perf_indices = {}
        self.sequences = sequences
        self.augment = augment
        self.piece_names = piece_names
        self.interpol_c2o = interpol_c2o
        self.staff_coords = staff_coords
        self.add_per_staff = add_per_staff

        self.fps = FPS
        self.sample_rate = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_SIZE

        self.length = len(self.sequences)
        self.gt_width = 30
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        seq = self.sequences[item]

        piece_id = seq['piece_id']
        score = self.scores[piece_id]

        is_onset = seq['is_onset']

        signal = self.performances[piece_id]

        # if signal is provided as a path it should be loaded from the disk
        if isinstance(signal, str):
            signal = load_wav(signal, SAMPLE_RATE)

        start_frame = int(seq['start_frame'])
        frame = int(seq['frame'])
        scale_factor = seq['scale_factor']

        start_t = int(start_frame * self.hop_length)
        t = self.frame_size + int(frame * self.hop_length)

        truncated_signal = signal[start_t:t]

        true_position, page_nr = seq['true_position'][:2], seq['true_position'][-1]

        max_y_shift = seq['max_y_shift']
        max_x_shift = seq['max_x_shift']

        page_nr = int(page_nr)

        s = score[page_nr]

        true_pos = np.copy(true_position)

        true_pos = true_pos / scale_factor
        width = self.gt_width / scale_factor
        height = seq['height'] / scale_factor

        if self.augment:

            yshift = random.randint(int(max_y_shift[0]/scale_factor), int(max_y_shift[1]/scale_factor))
            xshift = random.randint(int(max_x_shift[0] / scale_factor), int(max_x_shift[1] / scale_factor))

            true_pos[0] += yshift
            true_pos[1] += xshift

            s = np.roll(s, yshift, 0)
            s = np.roll(s, xshift, 1)

            # pad signal randomly by 0-20 frames (0-1seconds)
            truncated_signal = np.pad(truncated_signal, (random.randint(0, int(self.fps)) * self.hop_length, 0),
                                      mode='constant')

        center_y, center_x = true_pos

        target = np.asarray([[0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0]]],
                            dtype=np.float32)

        unscaled_targets = np.copy(target)
        unscaled_targets[:, 2] *= s.shape[1]
        unscaled_targets[:, 3] *= s.shape[0]
        unscaled_targets[:, 4] *= s.shape[1]
        unscaled_targets[:, 5] *= s.shape[0]

        unscaled_targets[:, 2:] *= scale_factor

        interpol_c2o = self.interpol_c2o[piece_id][page_nr]
        add_per_staff = [self.staff_coords[piece_id][page_nr], self.add_per_staff[piece_id][page_nr]]
        piece_name = f"{self.piece_names[piece_id]}_page_{page_nr}"

        sample = {'performance': truncated_signal,  'score': s[None], 'target': target,
                  'file_name': piece_name, 'is_onset': is_onset, 'interpol_c2o': interpol_c2o,
                  'add_per_staff': add_per_staff, 'scale_factor': scale_factor, 'unscaled_target': unscaled_targets}

        if self.transform:
            sample = self.transform(sample)

        return sample


class CustomBatch:
    def __init__(self, batch):
        self.file_names = [x['file_name'] for x in batch]
        self.perf = [torch.as_tensor(x['performance'], dtype=torch.float32) for x in batch]
        targets = []
        unscaled_targets = []
        for i, x in enumerate(batch):
            # add image idx to targets for loss computation
            if x['target'] is not None:
                target = x['target']

                target[:, 0] = i
                targets.append(target)

                unscaled_target = x['unscaled_target']
                unscaled_target[:, 0] = i
                unscaled_targets.append(unscaled_target)

        self.targets = torch.as_tensor(np.concatenate(targets), dtype=torch.float32)
        self.unscaled_targets = torch.as_tensor(np.concatenate(unscaled_targets), dtype=torch.float32)

        self.interpols = [x['interpol_c2o'] for x in batch]
        self.add_per_staff = [x['add_per_staff'] for x in batch]

        self.scores = torch.as_tensor(np.stack([x['score'] for x in batch]), dtype=torch.float32)
        self.scale_factors = torch.FloatTensor([x['scale_factor'] for x in batch]).float().unsqueeze(-1)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.scores = self.scores.pin_memory()
        self.perf = [p.pin_memory() for p in self.perf]
        self.targets = self.targets.pin_memory()
        self.unscaled_targets = self.unscaled_targets.pin_memory()
        self.scale_factors = self.scale_factors.pin_memory()
        return self


def collate_wrapper(batch):
    return CustomBatch(batch)


def compute_batch_stats(detections, true_positions, piece_stats, file_names, file_interpols, file_add_per_staff):
    gt = true_positions.float()[:, 2:4].cpu()
    pred = detections[:, :2].detach().cpu()

    for num, fname in enumerate(file_names):

        if fname not in piece_stats:
            piece_stats[fname] = {}

        if 'frame_diff' not in piece_stats[fname]:
            piece_stats[fname]['frame_diff'] = []

        staff_coords, add_per_staff = file_add_per_staff[num]

        staff_id_pred = np.argwhere(min(staff_coords, key=lambda y: abs(y - pred[num][1])) == staff_coords).item()
        staff_id_gt = np.argwhere(min(staff_coords, key=lambda y: abs(y - gt[num][1])) == staff_coords).item()

        # unroll x coord
        x_coord_gt = gt[num][0] + add_per_staff[staff_id_gt]
        x_coord_pred = pred[num][0] + add_per_staff[staff_id_pred]

        # calculate difference of onset frames
        frame_diff = abs(file_interpols[num](x_coord_pred) - file_interpols[num](x_coord_gt))

        piece_stats[fname]['frame_diff'].append(frame_diff)

    return piece_stats


def load_dataset(path, augment=False, scale_width=416, split_file=None, ir_path=None,
                 only_onsets=False, load_audio=True):

    scores = {}
    piece_names = {}
    all_sequences = []
    performances = {}
    interpol_c2os = {}
    staff_coords_all = {}
    add_per_staff_all = {}
    params = []

    if split_file is not None:

        split = load_yaml(split_file)
        files = [os.path.join(path, f'{file}.npz') for file in split['files']]

    else:
        files = glob.glob(os.path.join(path, '*.npz'))

    for i, score_path in enumerate(files):
        params.append(dict(
            i=i,
            piece_name=os.path.basename(score_path)[:-4],
            path=os.path.dirname(score_path),
            scale_width=scale_width,
            load_audio=load_audio
        ))

    print(f'Loading {len(params)} file(s)...')

    # results = [load_piece_sequences(params[0])]
    with get_context("fork").Pool(8) as pool:
        results = list(tqdm(pool.imap_unordered(load_sequences, params), total=len(params)))

        for result in results:
            i, score, signals, piece_name, sequences, interpol_c2o, staff_coords, add_per_staff = result
            scores[i] = score
            performances[i] = signals
            piece_names[i] = piece_name
            interpol_c2os[i] = interpol_c2o
            staff_coords_all[i] = staff_coords
            add_per_staff_all[i] = add_per_staff

            all_sequences.extend([seq for seq in sequences if (seq['is_onset'] or not only_onsets)])

    print('Done loading.')

    if ir_path is not None:
        print('Using Impulse Response Augmentation')
        ir_aug = ImpulseResponse(ir_paths=ir_path, ir_prob=0.5)
        transform = torchvision.transforms.Compose([ir_aug])
    else:
        transform = None

    return SequenceDataset(scores, performances, all_sequences, piece_names, interpol_c2os,
                           staff_coords_all, add_per_staff_all, augment=augment, transform=transform)


def iterate_dataset(network, dataloader, criterion, optimizer=None, clip_grads=None,
                    device=torch.device('cuda'), tempo_aug=False):
    train = optimizer is not None
    losses = {}

    piece_stats = {}

    if is_main_process():
        progress_bar = tqdm(total=len(dataloader), ncols=80)

    for batch_idx, data in enumerate(dataloader):

        scores = data.scores.to(device, non_blocking=True)
        scale_factors = data.scale_factors.to(device, non_blocking=True)
        targets = data.targets.to(device, non_blocking=True)

        perf = [p.to(device, non_blocking=True) for p in data.perf]

        with torch.set_grad_enabled(train):
            inference_out, pred = network(score=scores, perf=perf, tempo_aug=tempo_aug)

            loss_dict = criterion(pred, targets, network)
            loss = loss_dict['loss']
            for key in loss_dict:

                if key not in losses:
                    losses[key] = AverageMeter()

                losses[key].update(loss_dict[key].item())

        # perform update
        if train:
            optimizer.zero_grad()
            loss.backward()

            if clip_grads is not None:

                # only clip gradients of the recurrent network
                if hasattr(network, "conditioning_network"):
                    clip_grad_norm_(network.conditioning_network.seq_model.parameters(), clip_grads)
                else:
                    # distributed data parallel
                    clip_grad_norm_(network.module.conditioning_network.seq_model.parameters(), clip_grads)

            optimizer.step()

        unscaled_targets = data.unscaled_targets.to(device, non_blocking=True)

        inference_out = inference_out.detach()

        pred_boxes = get_max_box(inference_out)
        pred_boxes *= scale_factors

        piece_stats = compute_batch_stats(pred_boxes, unscaled_targets,
                                          piece_stats, data.file_names, data.interpols, data.add_per_staff)

        if is_main_process():
            progress_bar.update(1)

    # summarize statistics
    stats = {'piece_stats': {}}

    frame_diffs = []
    for key in piece_stats:
        stat = piece_stats[key]

        assert key not in stats['piece_stats']
        stats['piece_stats'][key] = stat['frame_diff']

        frame_diffs.extend(stat['frame_diff'])

    stats['frame_diffs_mean'] = float(np.mean(frame_diffs))

    # add losses to statistics
    for key in losses:
        stats[key] = losses[key].avg

    if is_main_process():
        progress_bar.close()

    return stats
