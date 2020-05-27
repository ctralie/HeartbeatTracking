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


def get_frames_bbox(frames):
    """
    Get the union of all bounding boxes in all of the frames
    Parameters
    ----------
    frames: list of MorphableFace
        The frames
    """
    bbox = np.zeros(4)
    if len(frames) > 0:
        bbox = frames[0].get_bbox()
        for f in frames[1::]:
            bbox = bbox_union(bbox, f.get_bbox())
    return bbox
    

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
    files = get_video_filepaths(path)
    files = files[0:win]
    if len(files) < win:
        print("Not enough files to fill first window")
        return
    istart = 0
    print("Loading window...")
    frames = [MorphableFace(files[i]) for i in range(win)]
    print("Computing facial landmarks...")
    for f in frames:
        tic = time.time()
        f.get_face_keypts()
        print("Elapsed time ", time.time()-tic)
    while istart + win <= len(files):
        ## Step 2: Setup common bounding box for all frames
        print("Setting up grids...")
        bbox = get_frames_bbox(frames)
        for f in frames:
            f.setup_grid(bbox)

        ## Step 3: Warp all frames to the first frame
        print("Warping frames...")
        f0 = frames[0]
        images = [f0.img]
        for f in frames[1::]:
            images.append(f.get_forward_map(f0.XKey))
        plt.figure(figsize=(12, 6))
        for i, (f, img) in enumerate(zip(frames, images)):
            plt.clf()
            plt.subplot(121)
            plt.imshow(f.img)
            plt.title("Frame {}".format(i))
            plt.subplot(122)
            plt.imshow(img)
            plt.savefig("{}.png".format(i))

        ## Slide window to the right
        istart += hop
        if istart + win <= len(files):
            print("Loading window...")
            # Shift frames over and load in the frames for the next loop
            frames[0:win-hop] = frames[hop::]
            for i in range(hop):
                frames[win-hop+i] = MorphableFace(files[istart+i])

track_heartbeat("CVPR2014Data/1", 10*30, 10)