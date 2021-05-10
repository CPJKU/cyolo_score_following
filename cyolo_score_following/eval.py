 
import argparse
import torch

import numpy as np

from cyolo_score_following.dataset import load_dataset, collate_wrapper, iterate_dataset
from cyolo_score_following.utils.data_utils import FPS
from cyolo_score_following.models.yolo import load_pretrained_model
from torch.utils.data import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--param_path', help='path to the stored network', type=str)
    parser.add_argument('--test_dir', help='path to test dataset.',)
    parser.add_argument('--only_onsets', help='only evaluate onset frames', default=False, action='store_true')
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--split_file', help='split file to only evaluate a subset from the test dirs',
                        default=None)
    parser.add_argument('--scale_width', help='sheet image scale factor', type=float, default=416)
    parser.add_argument('--num_workers', default=4, type=int, help="number of parallel datapool worker")
    parser.add_argument('--load_audio', default=False, action='store_true', help="preload audio files for datapool")
    parser.add_argument('--print_piecewise', default=False, action='store_true', help="print statistics for each piece")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network, criterion = load_pretrained_model(args.param_path)

    print(network)
    print(f"Putting model to {device}")
    network.to(device)

    network.eval()

    dataset = load_dataset(args.test_dir, augment=False, scale_width=args.scale_width, split_file=args.split_file,
                           only_onsets=args.only_onsets, load_audio=args.load_audio)

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_wrapper, pin_memory=True)

    stats = iterate_dataset(network, dataloader, optimizer=None, criterion=criterion, device=device)

    ordering = []
    max_str_len = 0
    for piece in stats['piece_stats'].keys():
        ordering.append((piece, np.mean(stats['piece_stats'][piece])))

        # store maximum string length for printing
        str_len = len(piece)
        if str_len > max_str_len:
            max_str_len = str_len

    ordering = sorted(ordering, key=lambda k: k[1], reverse=False)

    thresholds = [0.05, 0.1, 0.5, 1.0, 5.0]

    if args.print_piecewise:
        print("Piecewise frame tracking ratios")
        for piece, _ in ordering:

            piece_stat = stats['piece_stats'][piece]

            diffs = np.array(piece_stat)
            diffs = diffs / FPS
            total = len(diffs)

            cumulative_percentage = []
            for th in thresholds:
                cumulative_percentage.append(np.round(100 * np.sum(diffs <= th) / total, 1))

            print(f"{piece+':':<{max_str_len+1}}\t\t {cumulative_percentage}")
        print()

    frame_diffs = np.concatenate([piece_stats for piece_stats in stats['piece_stats'].values()]) / FPS
    total_frames = len(frame_diffs)

    ratio_str = ""

    print('Average frame tracking ratios:')
    for th in thresholds:
        ratio = np.sum(frame_diffs <= th) / total_frames
        percentage = np.round(100 * ratio, 1)

        ratio_str += f"& {ratio:.3f} "
        print(f'<= {th}: {percentage}')

    # string for latex table
    ratio_str += "\\\\"
    print(ratio_str)
