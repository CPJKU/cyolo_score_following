
import argparse
import os
import torch

import numpy as np

from cyolo_score_following.dataset import load_dataset, collate_wrapper, iterate_dataset, CLASS_MAPPING
from cyolo_score_following.utils.data_utils import FPS
from cyolo_score_following.models.yolo import load_pretrained_model
from torch.utils.data import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--param_path', help='path to the stored network', type=str)
    parser.add_argument('--test_dirs', help='path to test dataset.', nargs='+')
    parser.add_argument('--only_onsets', help='only evaluate onset frames', default=False, action='store_true')
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--split_files', help='split file to only evaluate a subset from the test dirs',
                        default=None, nargs='+')
    parser.add_argument('--scale_width', help='sheet image scale factor', type=float, default=416)
    parser.add_argument('--num_workers', default=4, type=int, help="number of parallel datapool worker")
    parser.add_argument('--load_audio', default=False, action='store_true', help="preload audio files for datapool")
    parser.add_argument('--print_piecewise', default=False, action='store_true', help="print statistics for each piece")
    parser.add_argument('--save_tag', default=None)
    parser.add_argument('--save_dir', type=str, default="")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network, criterion = load_pretrained_model(args.param_path)

    predict_sb = network.nc == 3
    print(network)
    print(f"Putting model to {device}")
    network.to(device)

    network.eval()

    dataset = load_dataset(args.test_dirs, augment=False, scale_width=args.scale_width, split_files=args.split_files,
                           only_onsets=args.only_onsets, load_audio=args.load_audio, predict_sb=predict_sb)

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_wrapper, pin_memory=True)

    stats = iterate_dataset(network, dataloader, optimizer=None, criterion=criterion, device=device)

    if args.save_tag is not None:
        with open(os.path.join(args.save_dir, args.save_tag + "_stats.npy"), "wb") as f:
            np.save(f, stats)

    ordering = []
    max_str_len = 0
    for piece in stats['piece_stats'].keys():
        ordering.append((piece, np.mean(stats['piece_stats'][piece]['frame_diff'])))

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
            print(f"{piece}:")
            if 'frame_diff' in piece_stat:
                diffs = np.array(piece_stat['frame_diff'])
                diffs = diffs / FPS
                total = len(diffs)

                cumulative_percentage = []
                for th in thresholds:
                    cumulative_percentage.append(np.round(100 * np.sum(diffs <= th) / total, 1))

                print("\tTracked Frame Ratios", cumulative_percentage)

            for value in CLASS_MAPPING.values():
                if value + "_accuracy" in piece_stat:
                    print(f"\t{value} Accuracy: {piece_stat[value + '_accuracy']:.3f}")

    print()
    for value in CLASS_MAPPING.values():
        if value + "_accuracy" in stats:
            print(f'Average accuracy for {value}: {stats[value + "_accuracy"]:.3f}')


    frame_diffs = np.concatenate([piece_stats['frame_diff'] for piece_stats in stats['piece_stats'].values()]) / FPS
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
