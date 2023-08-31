import os
import subprocess
from pathlib import Path

import numpy as np

rpi = 'raspberry' in os.uname().nodename
vcodec = ''# -vcodec h264_v4l2m2m' if 'raspberry' in os.uname().nodename else ''


class MVExtractor:
    def __init__(self):
        self.launch_cmd = f"ffmpeg -v error -f rawvideo -pix_fmt yuv420p -s 640x480 -i - {vcodec} -y -f h264 - | ffmpeg -v error -flags2 +export_mvs -f h264 -i -{vcodec} -filter printmvs -f null -"
        self.process2 = None
        self.restart_process()

    def restart_process(self):
        if self.process2 is None:
            self.process2 = subprocess.Popen(self.launch_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        self.process = self.process2
        self.process2 = subprocess.Popen(self.launch_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

    def extract(self, frame1, frame2):
        self.process.stdin.write(frame1.tobytes())
        self.process.stdin.write(frame2.tobytes())
        self.process.stdin.close()
        mvs = np.loadtxt(self.process.stdout.readlines(), dtype=np.float32, delimiter=",").reshape((-1, 4))
        self.restart_process()
        mvs = mvs[mvs[:, 1] != mvs[:, 3]]
        return mvs

args_print_mvs = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '640x480', '-i', 'pipe:',
                  '-y', '/tmp/out.h264']
in_file_print_mvs = subprocess.Popen(args_print_mvs, stdin=subprocess.PIPE)

def ffmpeg_encode_extract_printmvs(frame1: np.ndarray, frame2: np.ndarray):
    """
    Returns motion-vectors that were generated using FFmpeg.

    :param frame1: A frame as a Numpy array.
    :param frame2: A frame as a Numpy array.
    :param temp_dir: A directory to save temporary files to.
    """
    global in_file_print_mvs
    # args = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '640x480', '-i', 'pipe:', '-vcodec', 'h264_v4l2m2m', '-y', f'{temp_dir}/out.h264']
    in_file_print_mvs.stdin.write(frame1.tobytes())
    in_file_print_mvs.stdin.write(frame2.tobytes())
    in_file_print_mvs.stdin.close()
    in_file_print_mvs.wait()
    extractor = (ffmpeg.input("/tmp/out.h264", flags2='+export_mvs').filter('printmvs')
                 .output("-", format="null")
                 .run_async(quiet=True, pipe_stdout=True))
    in_file_print_mvs = subprocess.Popen(args_print_mvs, stdin=subprocess.PIPE)
    mvs = np.loadtxt(extractor.stdout.readlines(), dtype=np.float32, delimiter=",").reshape((-1, 4))
    mvs = mvs[mvs[:, 1] != mvs[:, 3]]
    return mvs
def ffmpeg_encode(frame, direct=False):
    if direct:
        launch_cmd = f"ffmpeg -v error -f rawvideo -pix_fmt yuv420p -s 640x480 -i - {vcodec} -y -f h264 - | ffmpeg -v error -flags2 +export_mvs -f h264 -i -{vcodec} -filter calcdepth=dH=30:fx=500:cx=320:fy=500:cy=240 -f null -"
    else:
        launch_cmd = f"ffmpeg -v error -f rawvideo -pix_fmt yuv420p -s 640x480 -i - {vcodec} -y -f h264 - | ffmpeg -v error -flags2 +export_mvs -f h264 -i -{vcodec} -filter printmvs -f null -"
    proc = subprocess.Popen(launch_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    return proc, frame


def ffmpeg_extract(encoder, frame, direct=False):
    encoder[0].stdin.write(encoder[1].tobytes())
    encoder[0].stdin.write(frame.tobytes())
    encoder[0].stdin.close()
    if direct:
        mvs = np.loadtxt(encoder[0].stdout.readlines(), dtype=np.float32, delimiter=",").reshape((-1, 3))*4 # still no idea why 4
    else:
        mvs = np.loadtxt(encoder[0].stdout.readlines(), dtype=np.float32, delimiter=",").reshape((-1, 4))
        mvs = mvs[mvs[:, 1] != mvs[:, 3]]
    return mvs


def ffmpeg_encode_extract(frame1: np.ndarray, frame2: np.ndarray):
    """
    Returns motion-vectors that were generated using FFmpeg.
    :param frame1: A frame as a Numpy array.
    :param frame2: A frame as a Numpy array.
    :param temp_dir: A directory to save temporary files to.
    """
    args = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '640x480', '-i', 'pipe:', '-y',
            '/tmp/out.h264']
    # args = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '640x480', '-i', 'pipe:', '-vcodec', 'h264_v4l2m2m', '-y', '/tmp/out.h264']
    in_file = subprocess.Popen(args, stdin=subprocess.PIPE)
    in_file.stdin.write(frame1.tobytes())
    in_file.stdin.write(frame2.tobytes())
    in_file.stdin.close()
    in_file.wait()
    args = [f'{Path(__file__).parent}/extract_mvs', "/tmp/out.h264"]
    extractor = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    mvs = np.frombuffer(extractor.stdout.read(), dtype=np.float32).reshape((-1, 4))
    mvs = mvs[mvs[:, 1] != mvs[:, 3]]
    return mvs


def ffmpeg_encode_extract_depth(frame1: np.ndarray, frame2: np.ndarray, cam_mat, height_diff):
    """
    Returns motion-vectors that were generated using FFmpeg.

    :param frame1: A frame as a Numpy array.
    :param frame2: A frame as a Numpy array.
    :param temp_dir: A directory to save temporary files to.
    """
    launch_cmd = f"ffmpeg -v error -f rawvideo -pix_fmt yuv420p -s 640x480 -i - {vcodec} -y -f h264 - | ffmpeg -v error -flags2 +export_mvs -f h264 -i -{vcodec} -filter calcdepth=dH={height_diff}:fx={cam_mat[0, 0]}:cx={cam_mat[0, 2]}:fy={cam_mat[1, 1]}:cy={cam_mat[1, 2]} -f null -"
    extractor = subprocess.Popen(launch_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    extractor.stdin.write(frame1.tobytes())
    extractor.stdin.write(frame2.tobytes())
    extractor.stdin.close()
    mvs = np.loadtxt(extractor.stdout.readlines(), dtype=np.float32, delimiter=",").reshape((-1, 3)) * 4  ## why times 4??
    return mvs


def ffmpeg_encode_extract_epzs(frame1: np.ndarray, frame2: np.ndarray):
    """
    Returns motion-vectors that were generated using FFmpeg.

    :param frame1: A frame as a Numpy array.
    :param frame2: A frame as a Numpy array.
    """
    args = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '640x480', '-i', 'pipe:', '-filter',
            'mestimate=epzs:mb_size=32:search_param=700,printmvs', '-f', 'null', '-']
    estimator = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    estimator.stdin.write(frame1.tobytes())
    estimator.stdin.write(frame2.tobytes())
    estimator.stdin.close()
    estimator.wait()
    mvs = np.loadtxt(estimator.stdout.readlines(), dtype=np.float32, delimiter=",").reshape((-1, 4))
    mvs = mvs[mvs[:, 1] != mvs[:, 3]]
    x_diff = np.abs(mvs[:, 0]-mvs[:, 2])
    mvs = mvs[x_diff < 40]
    return mvs
