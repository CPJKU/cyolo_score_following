
import cv2
import torch

import numpy as np

from cyolo_score_following.dataset import CLASS_MAPPING
from cyolo_score_following.models.yolo import load_pretrained_model
from cyolo_score_following.utils.data_utils import load_piece_for_testing, SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from cyolo_score_following.utils.general import xywh2xyxy
from cyolo_score_following.utils.video_utils import create_video, prepare_spec_for_render, plot_box, plot_line
from tqdm import tqdm


COLORS = [(0, 0, 1), (0, 0.5, 1), (0, 1, 0.5)]
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Network Test Video')
    parser.add_argument('--param_path', help='path to the stored network', type=str)
    parser.add_argument('--test_dir', help='path to test dataset.', type=str)
    parser.add_argument('--test_piece', help='name of test piece (do not specify extension).', type=str)
    parser.add_argument('--scale_width', help='sheet image scale factor.', type=int, default=416)
    parser.add_argument('--plot', help='intermediate plotting', default=False, action='store_true')
    parser.add_argument('--gt_only', help='only plot ground truth', default=False, action='store_true')
    parser.add_argument('--page', help='only track given page (start indexing at 0)', type=int, default=None)
    args = parser.parse_args()

    piece_name = args.test_piece
    org_scores, score, signal_np, systems, bars, interpol_fnc, pad, scale_factor, onsets = load_piece_for_testing(args.test_dir, piece_name, args.scale_width)

    if not args.gt_only:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network, criterion = load_pretrained_model(args.param_path)

        print(network)
        print("Putting model to %s ..." % device)
        network.to(device)
        print("Number of parameters:", sum(p.numel() for p in network.parameters() if p.requires_grad))
        network.eval()

        signal = torch.from_numpy(signal_np).to(device)
        score_tensor = torch.from_numpy(score).unsqueeze(1).to(device)

    from_ = 0
    to_ = FRAME_SIZE

    hidden = None
    observation_images = []
    frame_idx = 0

    actual_page = 0
    track_page = args.page
    start_ = None
    vis_spec = None

    pbar = tqdm(total=signal_np.shape[-1])

    while to_ <= signal_np.shape[-1]:
        true_position = np.array(interpol_fnc(frame_idx), dtype=np.float32)
        actual_system = int(true_position[2])
        actual_bar = int(true_position[3])
        if actual_page != int(true_position[-1]):
            hidden = None

        actual_page = int(true_position[-1])
        system = systems[actual_system]
        bar = bars[actual_bar]
        true_position = true_position[:2]

        if track_page is None or actual_page == track_page:
            start_ = from_ if start_ is None else start_

            if not args.gt_only:
                with torch.no_grad():

                    sig_excerpt = signal[from_:to_]
                    spec_frame = network.compute_spec([sig_excerpt], tempo_aug=False)[0]

                    z, hidden = network.conditioning_network.get_conditioning(spec_frame, hidden=hidden)
                    inference_out, pred = network.predict(score_tensor[actual_page:actual_page+1], z)

                x1, y1, x2, y2 = [], [], [], []
                for class_idx in range(network.nc):
                    filtered_inference_out = inference_out[
                        # 0, inference_out[0, :, 5:].argmax(-1) == class_idx].unsqueeze(0)
                        0, inference_out[0, :, -1] == class_idx].unsqueeze(0)

                    _, idx = torch.sort(filtered_inference_out[0, :, 4], descending=True)
                    filtered_pred = filtered_inference_out[0, idx[:1]]
                    box = filtered_pred[..., :4]
                    conf = filtered_pred[..., 4]

                    x1_, y1_, x2_, y2_ = xywh2xyxy(box).cpu().numpy().T
                    x1.append(x1_)
                    y1.append(y1_)
                    x2.append(x2_)
                    y2.append(y2_)

                x1 = np.concatenate(x1) * scale_factor - pad
                x2 = np.concatenate(x2) * scale_factor - pad
                y1 = np.concatenate(y1) * scale_factor
                y2 = np.concatenate(y2) * scale_factor

                if vis_spec is not None:
                    vis_spec = np.roll(vis_spec, -1, axis=1)
                else:
                    vis_spec = np.zeros((spec_frame.shape[-1], 40))

                vis_spec[:, -1] = spec_frame[0].cpu().numpy()

            height = system['h'] / 2
            center_y, center_x = true_position

            img_pred = cv2.cvtColor(org_scores[actual_page], cv2.COLOR_RGB2BGR)

            plot_line([center_x - pad, center_y, height], img_pred, label="GT",
                      color=(0.96, 0.63, 0.25), line_thickness=2)

            if center_x > 0:
                plot_line([center_x - pad, center_y, height], img_pred, label="GT Note", color=(0.96, 0.63, 0.25), line_thickness=2)

            plot_box(xywh2xyxy(np.asarray([[system['x'] - pad, system['y'],
                                                system['w'], system['h']]]))[0].astype(int).tolist(),
                         img_pred, color=(0.25, 0.71, 0.96), line_thickness=2, label="GT System")

            plot_box(xywh2xyxy(np.asarray([[bar['x'] - pad, bar['y'],
                                                bar['w'], bar['h']]]))[0].astype(int).tolist(),
                         img_pred, color=(0.96, 0.24, 0.69), line_thickness=2, label="GT Bar")

            if not args.gt_only:
                for i in range(network.nc-1, -1, -1):
                    pred_label = f"Pred {CLASS_MAPPING[i]}"
                    plot_box([x1[i], y1[i], x2[i], y2[i]], img_pred, label=pred_label,
                                 color=COLORS[i % len(COLORS)], line_thickness=2)

                perf_img = prepare_spec_for_render(vis_spec, img_pred)
            else:
                perf_img = np.zeros((img_pred.shape[0], 200, 3))

            img = np.concatenate((img_pred, perf_img), axis=1)
            img = np.array((img*255), dtype=np.uint8)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            line_type = 2

            if args.plot:

                cv2.imshow('Prediction', img)
                cv2.waitKey(20)

            observation_images.append(img)
        else:
            if start_ is not None:
                # avoid moving back to the page
                # (in case repetitions span across multiple pages, shouldn't happen in msmd)
                break

        from_ += HOP_SIZE
        to_ += HOP_SIZE
        frame_idx += 1

        pbar.update(HOP_SIZE)

    pbar.close()

    truncated_signal = signal_np[start_:to_]
    tag = "" if args.page is None else f"_{args.page}"

    create_video(observation_images, truncated_signal, piece_name, FPS, SAMPLE_RATE, tag=tag, path="../videos")
