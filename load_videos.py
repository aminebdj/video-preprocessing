import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import cv2
from tqdm import tqdm
from util import save
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
import torchaudio
warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-subs', '--write-subs',
                     '--sub-langs', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path

def run(data):
    video_id, args = data
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
       download(video_id.split('#')[0], args)

    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
       print ('Can not load video %s, broken link' % video_id.split('#')[0])
       return 
    # reader = imageio.get_reader(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4'))
    pth = os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')
    cap = cv2.VideoCapture(pth)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the audio tensor
    # try:
    audio_tensor, samplerate = torchaudio.load(pth)
    audio_frames = audio_tensor.shape[1]
    # except:
    #     print("Error loading audio for video %s" % video_id)
    #     return

    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]
    
    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames':[], 'audio': []} for j in range(df.shape[0])]

    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]
    i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for entry in all_chunks_dict:
                if (i * fps >= entry['start'] * fps) and (i * fps < entry['end'] * fps):
                    left, top, right, bot = entry['bbox']
                    left = int(left / (ref_width / frame.shape[1]))
                    top = int(top / (ref_height / frame.shape[0]))
                    right = int(right / (ref_width / frame.shape[1]))
                    bot = int(bot / (ref_height / frame.shape[0]))
                    crop = frame[top:bot, left:right]
                    if args.image_shape is not None:
                       crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))
                    entry['frames'].append(crop)
            i += 1
        cap.release()
    except imageio.core.format.CannotReadFrameError:
        None

    for entry in all_chunks_dict:
        if 'person_id' in df:
            first_part = df['person_id'].iloc[0] + "#"
        else:
            first_part = ""
        first_part = first_part + '#'.join(video_id.split('#')[::-1])
        path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'
        start_audio, end_audio = int((entry['start'] / video_frames) * audio_frames), int((entry['end'] / video_frames) * audio_frames)
        subclip_audio = audio_tensor[:,int(start_audio):int(end_audio)]
        save(os.path.join(args.out_folder, partition, path), entry['frames'], subclip_audio, args.format, fps, samplerate, args.desired_fps, args.desired_sample_rate)
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='youtube-taichi', help='Path to youtube videos')
    parser.add_argument("--metadata", default='taichi-metadata.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='taichi-png', help='Path to output')
    parser.add_argument("--format", default='.png', help='Storing format')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--youtube", default='yt-dlp', help='Path to youtube-dl')
    parser.add_argument("--desired_fps", default=25, type=int, help='Desired FPS')
    parser.add_argument("--desired_sample_rate", default=16000, type=int, help='Desired sample rate')
    parser.add_argument("--image_shape", default=(512, 512), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None for no resize")

    args = parser.parse_args()
    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        None  
