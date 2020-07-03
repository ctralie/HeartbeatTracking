from FaceTools import *
import scipy.io as sio
from scipy.signal import sosfilt, sosfiltfilt, butter, periodogram
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

def bandpass_filter_timeseries(y, fs, nfilt = 10, fl=0.5, fh=5):
    """
    Apply a zero-phase Butterworth bandpass filter
    Parameters
    ----------
    y: ndarray(N)
        The signal to filter
    fs: int
        The samplerate
    nfilt: int
        Order of the bandpass filter
    fl: float
        Left cutoff of bandpass filter (in hz)
    fr: float
        Right cutoff of bandpass filter (in hz)
    """
    minfq = 2*fl/fs
    maxfq = 2*fh/fs
    sos = butter(nfilt, [minfq, maxfq], btype='bandpass', output='sos')
    yfilt = sosfiltfilt(sos, y)
    return yfilt

def aggregate_kumar(Y, fs, Ath = 8, fl=0.5, fh=5, bwin=10, fac = 10):
    """
    Aggregate all of the time series from spatial blocks using
    the technique from
    [1] Kumar, Mayank, Ashok Veeraraghavan, and Ashutosh Sabharwal. 
        "DistancePPG: Robust non-contact vital signs monitoring using a camera." 
        Biomedical optics express 6.5 (2015): 1565-1588.
    Parameters
    ----------
    Y: list of ndarray(M, N)
        All of the different time series
    fs: int
        Sample rate
    Ath: int
        Range cutoff for a block
    fl: float
        Left cutoff of bandpass filter (in hz)
    fr: float
        Right cutoff of bandpass filter (in hz)
    bwin: float
        Half-length of interval to integrate for goodness
        of fit, in beats per minute
    fac: int
        Factor by which to up-sample the PSD
    Returns
    -------
    {
        'yest': ndarray(N)
            Initial time series estimate,
        'yfinal': ndarray(N)
            Final time series estimate,
        'goodness': ndarray(M)
            Goodness of fits,
        'bpmorig': float
            Initial beats per minute estimate,
        'bpm': float
            Final beats per minute estimate
    }
    """
    # First, we assume that all the Gi are equal to 1.0
    YRange = np.max(Y, axis=1) - np.min(Y, axis=1)
    Y = Y[YRange < Ath, :]
    yest = np.mean(Y, 0)
    # Come up with an initial estimate of the pulse
    freq, p = periodogram(yest, nfft=yest.size*fac)
    freq = freq*fs*60
    
    # Come up with goodness of fit ratios based on this estimate
    idx = np.argmax(p)
    bpmorig = freq[idx]
    idxnum = np.arange(freq.size)
    idxnum = idxnum[np.abs(freq[idx]-freq) <= bwin]
    idxdenom = np.arange(freq.size)
    idxdenom = idxdenom[(freq > fl*60)*(freq < fh*60)]
    freq, P = periodogram(Y, nfft=yest.size*fac, axis=1)
    num = np.sum(P[:,idxnum], axis=1)
    denom = np.sum(P[:, idxdenom], axis=1) - num
    goodness = num/denom
    goodness[goodness < 0] = 0

    # Compute final time series and pulse estimate
    yfinal = np.sum(goodness[:, None]*Y, 0)
    freq, p = periodogram(yfinal, nfft=yfinal.size*fac)
    bpm = freq[np.argmax(p)]*fs*60
    goodnessret = np.zeros(YRange.size)
    goodnessret[YRange < Ath] = goodness
    return {'yest':yest, 'yfinal':yfinal, 'goodness':goodnessret, 'bpm':bpm, 'bpmorig':bpmorig}


def track_heartbeat(path, win, hop, fs, block_size = 20, fine_points = False, show_warps=False, show_block_timeseries = False, plot_final_timeseries=False):
    """
    Parameters
    ---------
    path: string
        Path to a directory or video file
    win: int
        The number of frames in each window
    hop: int
        The number of frames to jump in between each window
    fs: int
        Sample rate of video
    block_size: int
        The dimension of each block that's averaged
    fine_points: boolean
        Whether to include fine scale good points to track
    show_warps: boolean
        Whether to save plots showing the warping
    show_block_timeseries: boolean  
        Whether to plot information about the time series in each block
    plot_final_timeseries: boolean
        Whether to plot the final time series and show goodness of 
        fit estimates
    """
    import cv2
    # Parameters for lucas kanade optical flow
    # From OpenCV tutorial https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    files = get_video_filepaths(path)
    if len(files) < win:
        print("Not enough files to fill first window")
        return
    istart = 0
    window_num = 1
    while istart + win <= len(files):
        ## Step 0: Load in images
        print("Loading block {}...".format(window_num))
        frames = [MorphableFace(files[i+istart]) for i in range(win)]

        ## Step 1: Track frames
        print("Tracking...")
        frames[0].get_face_keypts()
        if fine_points:
            p = frames[0].get_good_points_to_track()
            frames[0].add_keypts(p)
        frames[0].exclude_landmarks([LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS])
        last_img = cv2.cvtColor(frames[0].img, cv2.COLOR_RGB2GRAY)
        p0 = np.array(frames[0].XKey, dtype=np.float32)
        idx = np.arange(p0.shape[0])
        print("{} Points tracking".format(p0.shape[0]))
        ps = [p0]
        N = p0.shape[0]
        for i, f in enumerate(frames[1::]):
            # Documentation on optical flow:
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
            this_img = cv2.cvtColor(f.img, cv2.COLOR_RGB2GRAY)
            # calculate optical flow
            p1_idx, st, _ = cv2.calcOpticalFlowPyrLK(last_img, this_img, p0, None, **lk_params)
            # Select subset of points that are tracked well
            st = st.flatten()
            p1_idx = p1_idx[st == 1]
            idx = idx[st == 1]
            p1 = np.zeros((N, 2))
            p1[idx] = p1_idx
            ps.append(p1)
            # Now update the previous frame and previous points
            last_img = this_img.copy()
            p0 = p1_idx
        print("{} Points survived tracking".format(idx.size))
        for i, p in enumerate(ps):
            p = p[idx]
            frames[i].XKey = p

        ## Step 2: Setup common bounding box for all frames
        print("Setting up grids...")
        tic = time.time()
        bbox = get_frames_bbox(frames)
        for i, f in enumerate(frames):
            f.setup_grid(bbox)
        print("Elapsed Time", time.time()-tic)
            

        ## Step 3: Warp all frames to the first frame
        print("Warping frames...")
        tic = time.time()
        f0 = frames[0]
        images = [f0.img]
        if show_warps:
            plt.figure(figsize=(18, 6))
        for i, f in enumerate(frames[1::]):
            images.append(f.get_forward_map(f0.XKey))
            if show_warps:
                plt.clf()
                plt.subplot(131)
                f.plotKeypoints(drawTriangles=True)
                plt.xlim([bbox[2], bbox[3]])
                plt.ylim([bbox[1], bbox[0]])
                plt.subplot(132)
                plt.imshow(f.img)
                plt.title("Frame {}".format(i))
                plt.subplot(133)
                plt.imshow(images[-1])
                plt.savefig("{}.png".format(i))
        if show_warps:
            subprocess.call(["ffmpeg", "-i", "%d.png", "-b", "5000k", "{}.ogg".format(window_num)])
        # Extract green channel only and stack up into numpy array
        for i, image in enumerate(images):
            images[i] = image[:, :, 1]
        images = np.array(images)
        print("Elapsed Time", time.time()-tic)

        ## Step 4: Split up time series into blocks
        X = frames[0].get_blocks(block_size)
        X = get_block_pixel_indices(X, block_size)
        Y = [] # The time series
        if show_block_timeseries:
            plt.figure(figsize=(12, 6))
        for i in range(X.shape[0]):
            I, J = X[i, :, 0], X[i, :, 1]
            y = images[:, I, J]
            y = np.mean(y, axis=1)
            y = bandpass_filter_timeseries(y, fs)
            Y.append(y)
            image = np.array(images[0, :, :])[:, :, None]
            image = np.concatenate((image, image, image), axis=2)
            image[I, J, 1] = 1
            if show_block_timeseries:
                plt.clf()
                plt.subplot(131)
                plt.imshow(image)
                plt.subplot2grid((1, 3), (0, 1), colspan=2)
                plt.plot(y)
                plt.savefig("Win{}_Patch{}.png".format(window_num, i))
        Y = np.array(Y)

        ## Step 5: Do periodicity analysis within blocks
        res = aggregate_kumar(Y, fs)
        if plot_final_timeseries:
            image = np.array(images[0, :, :])[:, :, None]
            image = np.concatenate((image, image, image), axis=2)
            goodness = res['goodness']
            goodness = 255*goodness/np.max(goodness)
            for i, gi in enumerate(goodness):
                I, J = X[i, :, 0], X[i, :, 1]
                image[I, J, 0] = gi

            plt.figure(figsize=(12, 6))
            plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
            plt.imshow(image)
            plt.title("Goodness image")
            plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
            plt.plot(res['yest'])
            plt.title("Initial Estimate, %.3g BPM"%res['bpmorig'])
            plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=2)
            plt.plot(res['yfinal'])
            plt.title("Final Estimate, %.3g BPM"%res['bpm'])
            plt.savefig("Win{}Extimate.png".format(window_num))



        istart += hop
        window_num += 1

if __name__ == '__main__':
    #"BUData/T10_T11_30Subjects/F024/T11"
    win = 150
    hop = 25
    fs = 25
    track_heartbeat("BUData/first10subjects_2D/F013/T1", win, hop, fs, fine_points=True, plot_final_timeseries=True)