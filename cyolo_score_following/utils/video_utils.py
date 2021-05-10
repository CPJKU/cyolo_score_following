
import cv2
import os
import tempfile
import time
import random

import matplotlib.cm as cm
import numpy as np
import soundfile as sf


def write_video(images, fn_output='output.mp4', frame_rate=20, overwrite=False):
    """Takes a list of images and interprets them as frames for a video.

    Source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    """
    height, width, _ = images[0].shape

    if overwrite:
        if os.path.exists(fn_output):
            os.remove(fn_output)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fn_output, fourcc, frame_rate, (width, height))

    for cur_image in images:
        frame = cv2.resize(cur_image, (width, height))
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()

    return fn_output


def mux_video_audio(path_video, path_audio, path_output='output_audio.mp4'):
    """Use FFMPEG to mux video with audio recording."""
    from subprocess import check_call

    check_call(["ffmpeg", "-y", "-i", path_video, "-i", path_audio, "-shortest", "-c:v", "h264", path_output])


def create_video(observation_images, signal, piece_name, fps, sample_rate, path="../videos", tag=""):

    if not os.path.exists(path):
        os.mkdir(path)

    # create temp wavfile
    wav_path = os.path.join(tempfile.gettempdir(), str(time.time()) + '.wav')
    sf.write(wav_path, signal, samplerate=sample_rate)

    path_video = write_video(observation_images,
                             fn_output=os.path.join(tempfile.gettempdir(), str(time.time()) + '.mp4'),
                             frame_rate=fps, overwrite=True)

    # mux video and audio with ffmpeg
    mux_video_audio(path_video, wav_path, path_output=os.path.join(path, f'{piece_name}{tag}.mp4'))

    # clean up
    os.remove(path_video)


def prepare_spec_for_render(spec, score, scale_factor=5):
    spec_excerpt = cv2.resize(np.flipud(spec), (spec.shape[1] * scale_factor, spec.shape[0] * scale_factor))

    perf_img = np.pad(cm.viridis(spec_excerpt)[:, :, :3],
                      ((score.shape[0] // 2 - spec_excerpt.shape[0] // 2 + 1,
                        score.shape[0] // 2 - spec_excerpt.shape[0] // 2),
                       (20, 20), (0, 0)), mode="constant")

    return perf_img


def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_line(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    cx, cy, h = int(x[0]), int(x[1]), int(x[2])

    cv2.line(img, (cx, cy - h // 2), (cx, cy + h // 2), color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = cx + t_size[0], cy - h // 2 - t_size[1] - 3
        cv2.rectangle(img, (cx, cy - h // 2), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (cx, cy - h // 2 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
