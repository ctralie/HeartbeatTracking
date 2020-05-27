from FaceTools import *
from FundamentalFreq import *
import os
import glob
import subprocess

TEMP_FOLDERNAME = "tempvideo"

def get_video_filepaths(path):
    """
    Create a list of file paths corresponding to a video.
    If the path passed is a video file, then extract the frames
    to a folder and return the paths in that folder
    Parameters
    ----------
    path: string
        A path either to a directory of frames or to a video file
    Returns
    -------
    files: list of string
        List of file paths, sorted in time order
    """
    # First check to see if it's a video file or a folder
    foldername = path
    if not os.path.isdir(path):
        foldername = TEMP_FOLDERNAME
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        # Remove all files in this folder
        for f in glob.glob("{}{}*".format(foldername, os.path.sep)):
            os.remove(f)
        # Use ffmpeg to extract
        print("Extracting video...")
        subprocess.call(["ffmpeg", "-i", path, "-f", "image2", "{}{}%d.png".format(foldername, os.path.sep)])
        print("Finished extracting video")
    files = glob.glob("{}{}*".format(foldername, os.path.sep))
    # Find the most common extension (in case there are other non-frame
    # files in the directory)
    ext_count = {}
    for f in files:
        _, ext = os.path.splitext(f)
        if ext in ext_count:
            ext_count[ext] += 1
        else:
            ext_count[ext] = 1
    counts = np.array(ext_count.values())
    ext = list(ext_count.keys())[np.argmax(counts)]
    # Sort the files in time order
    pattern = "{}{}*{}".format(foldername, os.path.sep, ext)
    files = glob.glob(pattern)
    nums = []
    for f in files:
        prefix, _ = os.path.splitext(f)
        num = int(prefix.split(os.path.sep)[-1])
        nums.append(num)
    idxs = np.argsort(np.array(nums))
    files = [files[i] for i in idxs]
    return files


def track_heartbeat(path, win, hop):
    """
    Parameters
    ---------
    path: string
        Path to a directory or video file
    win: int
        The number of frames in each window
    hop: int
        The number of frames to jump in between each window
    """
    pass

