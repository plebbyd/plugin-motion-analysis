import cv2
import numpy as np
from typing import Dict
from Classifiers import TextureTemporalClassifier
import pickle
import time
import argparse
from collections import deque
from waggle.plugin import Plugin
from waggle.data.vision import Camera, VideoSample
from waggle.data.timestamp import get_timestamp
from pathlib import Path

import ffmpeg
import os

TOPIC_FLOWDETECTOR = 'env.flow.detection'

models = {50: 'tt_classifier_50fps.model',
          5: 'tt_classifier_5fps.model',
          1: 'tt_classifier_1fps.model'}

model_data = {rate: pickle.load(open(filename, 'rb')) for rate, filename in models.items()}
# model_data: Dict[TextureTemporalClassifier]

def get_water_mask(camera: Camera, model: TextureTemporalClassifier):
    N_FRAMES_MAX_BUFFER = 60
    test_framerate = time.time_ns()
    frame_buffer = deque(maxlen=N_FRAMES_MAX_BUFFER)
    for sample in camera.stream():
        frame_buffer.append(sample.data)
        if len(frame_buffer) == N_FRAMES_MAX_BUFFER:
            break
    time_to_accumulate_s = (float(time.time_ns()) - test_framerate) / (10**9)
    fps = 1.0 / (time_to_accumulate_s / N_FRAMES_MAX_BUFFER)
    print("FPS: %f" % fps)


def get_stream_info(stream_url):
    try:
        input_probe = ffmpeg.probe(stream_url)
        fps = eval(input_probe['streams'][0]['r_frame_rate'])
        width = int(input_probe['streams'][0]['width'])
        height = int(input_probe['streams'][0]['height'])
        return True, fps, width, height
    except:
        return False, 0., 0, 0


def take_sample(stream, duration, skip_second, resampling, resampling_fps):
    # Use Camera.record() to capture video
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = os.getcwd()
    filename_raw = os.path.join(script_dir, 'motion_record_raw.mp4')
    filename = os.path.join(script_dir, 'motion_record.mp4')
    
    camera = Camera(stream)
    video_sample = camera.record(duration=duration, file_path=filename_raw, skip_second=skip_second)
    timestamp = video_sample.timestamp

    # If resampling is needed, use ffmpeg to resample
    if resampling:
        print(f'Resampling to {resampling_fps}...')
        d = ffmpeg.input(filename_raw)
        d = ffmpeg.filter(d, 'fps', fps=resampling_fps)
        d = ffmpeg.output(d, filename, f='mp4', t=duration).overwrite_output()
        print(d.compile())
        d.run(quiet=True)
    else:
        # Just copy the file if no resampling needed
        d = ffmpeg.input(filename_raw)
        d = ffmpeg.output(d, filename, codec="copy", f='mp4', t=duration).overwrite_output()
        print(d.compile())
        d.run(quiet=True)
    
    return True, filename, timestamp



def run(args):
    # Initialize the plugin using context manager
    with Plugin() as plugin:
        logtimestamp = time.time()
        plugin.publish(TOPIC_FLOWDETECTOR, 'Flow Detector: Getting Video', timestamp=int(logtimestamp * 1e9))
        print(f"Getting Video: {logtimestamp}")

        # Get stream info using Camera
        camera = Camera(args.stream)
        device_url = camera.capture.device
        ret, fps, width, height = get_stream_info(device_url)
        if ret == False:
            print(f'Error probing {device_url}. Please make sure to put a correct video stream')
            return 1
        print(f'Input stream {device_url} with size of W: {width}, H: {height} at {fps} FPS')

        # If resampling is True, we use resampling_fps for inferencing as well as sampling
        if args.resampling:
            fps = args.resampling_fps
            print(f'Input will be resampled to {args.resampling_fps} FPS')


        sampling_countdown = -1
        if args.sampling_interval > -1:
            print(f'Input video will be sampled every {args.sampling_interval}th inferencing')
            sampling_countdown = args.sampling_interval

        logtimestamp = time.time()
        plugin.publish(TOPIC_FLOWDETECTOR, 'Flow Detector: Starting detector', timestamp=int(logtimestamp * 1e9))
        print('Starting flow detector..')
        while True:
            print(f'Grabbing video for {args.duration} seconds')
            ret, filename, timestamp = take_sample(
                stream=args.stream,
                duration=args.duration,
                skip_second=args.skip_second,
                resampling=args.resampling,
                resampling_fps=args.resampling_fps
            )
            if ret == False:
                print('Coud not sample video. Exiting...')
                return 1

            print('Analyzing the video...')
            total_frames = 0
            do_sampling = False
            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                do_sampling = True
                sampling_countdown = args.sampling_interval


            # Use VideoSample to read the recorded video
            video = VideoSample(path=filename, timestamp=timestamp)
            
            with video:
                width = video.capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                height = video.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
                fps = video.fps
                diff_time = 1. / 5.
                print(f'width: {width}, height: {height}, fps: {fps}')

                if do_sampling:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter("motionsample.mp4", fourcc, fps, (int(width), int(height)), True)

                c = 0
                input_frames = []
                t = 0
                for frame_sample in video:
                    t += 1. / fps
                    if t >= diff_time:
                        input_frames.append(frame_sample.data)
                        c += 1
                        t -= diff_time
                        if c == 60:
                            break

            input_frames = np.array(input_frames)
            input_frames = np.expand_dims(input_frames, axis=0)
            print(f'{input_frames.shape}')
            segmentation_array = model_data[5].segment(input_frames, prob_mode=True)
            #print(segmentation_array.shape)
            #print(segmentation_array.dtype)
            result = segmentation_array.squeeze()
            #print(result.shape)
            #print(result)


            logtimestamp = time.time()
            plugin.publish(TOPIC_FLOWDETECTOR, 'Flow Detector: End Detection', timestamp=int(logtimestamp * 1e9))
            print(f"End Detection: {logtimestamp}")

            # result *= 255.
            # print(f'{result}')
            # result[result > 0.] = 124.
            # result = result.astype(np.int8)
            result2 = cv2.cvtColor((result*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            #print(result2.dtype)
            cv2.imwrite('result.jpg', result2)
            print('saved')

            #count = 0
            #for i in range(len(result)):
            #    for j in range(len(result[0])):
            #        if result[i][j] > 0.7:
            #            count += 1
            #print(count)

            plugin.upload_file("motion_record.mp4")
            plugin.upload_file("result.jpg")

            logtimestamp = time.time()
            plugin.publish(TOPIC_FLOWDETECTOR, 'Flow Detector: End plugin', timestamp=int(logtimestamp * 1e9))
            print(f"End plugin: {logtimestamp}")
            exit(0)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera", type=str,
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-duration', dest='duration',
        action='store', default=14., type=float,
        help='Time duration for input video')
    parser.add_argument(
        '-resampling', dest='resampling', default=False,
        action='store_true', help="Resampling the sample to -resample-fps option (defualt 12)")
    parser.add_argument(
        '-resampling-fps', dest='resampling_fps',
        action='store', default=12, type=int,
        help='Frames per second for input video')
    parser.add_argument(
        '-skip-second', dest='skip_second',
        action='store', default=3., type=float,
        help='Seconds to skip before recording')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Inferencing interval for sampling results')
    args = parser.parse_args()
    run(args)
