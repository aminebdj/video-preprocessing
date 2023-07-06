import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize
import torchaudio
import imageio
import cv2, os

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def one_box_inside_other(boxA, boxB):
    xA = boxA[0] <= boxB[0]
    yA = boxA[1] <= boxB[1]
    xB = boxA[2] >= boxB[2]
    yB = boxA[3] >= boxB[3]
    return xA and yA and xB and yB

def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    return (left, top, right, bot)

def compute_increased_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    left = int(left - increase_area * width)
    top = int(top - increase_area * height)
    right = int(right + increase_area * width)
    bot = int(bot + increase_area * height)

    return (left, top, right, bot)

def crop_bbox_from_frames(frame_list, tube_bbox, min_frames=16, image_shape=(256, 256), min_size=200,
                          increase_area=0.1, aspect_preserving=True):
    frame_shape = frame_list[0].shape
    # Filter short sequences
    if len(frame_list) < min_frames:
        return None, None
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top
    # Filter if it is too small
    if max(width, height) < min_size:
        return None, None

    if aspect_preserving:
        left, top, right, bot = compute_aspect_preserved_bbox(tube_bbox, increase_area)
    else:
        left, top, right, bot = compute_increased_bbox(tube_bbox, increase_area)

    # Compute out of bounds
    left_oob = -min(0, left)
    right_oob = right - min(right, frame_shape[1])
    top_oob = -min(0, top)
    bot_oob = bot - min(bot, frame_shape[0])

    #Not use near the border
    if max(left_oob / float(width), right_oob / float(width), top_oob  / float(height), bot_oob / float(height)) > 0:
        return [None, None]

    selected = [frame[top:bot, left:right] for frame in frame_list]
    if image_shape is not None:
        out = [img_as_ubyte(resize(frame, image_shape, anti_aliasing=True)) for frame in selected]
    else:
        out = selected

    return out, [left, top, right, bot]

from multiprocessing import Pool
from itertools import cycle
from tqdm import tqdm
import os

def scheduler(data_list, fn, args):
    device_ids = args.device_ids.split(",")
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    f = open(args.chunks_metadata, 'w')
    line = "{video_id},{start},{end},{bbox},{fps},{width},{height},{partition}"
    print (line.replace('{', '').replace('}', ''), file=f)
    for chunks_data in tqdm(pool.imap_unordered(fn, zip(data_list, cycle(device_ids), args_list))):
        for data in chunks_data:
            print (line.format(**data), file=f)
            f.flush()
    f.close()

def write_frames_to_video(frame_list, output_file, fps, ffmpeg, remove, desired_fps):
    height, width, channels = frame_list[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec (e.g., 'XVID', 'MJPG', 'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in frame_list:
        video_writer.write(frame)

    # Release the video writer and destroy any remaining windows
    video_writer.release()
    os.system(ffmpeg.format(output_file, 'r', desired_fps, output_file.replace('.mp4', '-r.mp4')))
    os.system(remove.format(output_file))

def write_audio(audio_list, output_file, samplerate, ffmpeg, remove, desired_sample_rate):
    # write torchaudio tensor to wav file
    torchaudio.save(output_file, audio_list, samplerate)
    os.system(ffmpeg.format(output_file, 'ar', desired_sample_rate,  output_file.replace('.wav', '-r.wav'))) 
    os.system(remove.format(output_file))

def save(path, frames, audio, format, fps, samplerate, desired_fps, desired_sample_rate):
    ffmpeg_command = "ffmpeg -y -i {} -{} {} {}"
    remove_command = "rm {}"
    if format == '.mp4':
        write_frames_to_video(frames, path, fps, ffmpeg_command, remove_command,  desired_fps)
        # write_audio(audio, path.replace('.mp4', '.wav'), samplerate, ffmpeg_command, remove_command, desired_sample_rate)
        
    else:
        print ("Unknown format %s" % format)
        exit()