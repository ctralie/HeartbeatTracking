import time
import dlib
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import griddata, interpn
from GeometryTools import *

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

LEFT_EYE_LANDMARKS = np.arange(36, 42)
LEFT_EYEBROW_LANDMARKS = np.arange(17, 22)
RIGHT_EYE_LANDMARKS = np.arange(42, 48)
RIGHT_EYEBROW_LANDMARKS = np.arange(22, 27)
NOSE_LANDMARKS = np.arange(27, 36)
MOUTH_LANDMARKS = np.arange(48, 67)
EYEBROW_LANDMARKS = np.arange(17, 27)

def shape_to_np(shape, dtype="int"):
    """
    Used to convert from a shape object returned by dlib to an np array
    """
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=dtype)

class MorphableFace(object):
    """
    An object which stores a face along with facial landmarks, as computed by dlib,
    which can be used to warp the face into different expressions
    """
    def __init__(self, filename):
        """
        Constructor for the face object.  Loads in the image and saves it
        Parameters
        ----------
        filename: string
            Path to image file with at least one face in it
        """
        self.img = dlib.load_rgb_image(filename)

    def get_bbox(self):
        """
        Get the bounding box of the keypoints
        """
        j1, i1 = np.floor(np.min(self.XKey, axis=0))
        j2, i2 = np.ceil(np.max(self.XKey, axis=0))
        return clamp_bbox(np.array([i1, i2, j1, j2], dtype=int), self.img.shape)

    def setup_grid(self, bbox):
        """
        Setup a grid given a bounding box, and compute triangle
        indices and barycentric coordinates on this grid
        Parameters
        ----------
        bbox: ndarray([i1, i2, j1, j2])
            A bounding box
        """
        self.bbox = bbox
        self.pixx = np.arange(bbox[2], bbox[3]+1)
        self.pixy = np.arange(bbox[0], bbox[1]+1)
        self.XKeyWBbox = add_bbox_to_keypoints(self.XKey, bbox)
        self.tri = Delaunay(self.XKeyWBbox)
        X, Y = np.meshgrid(self.pixx, self.pixy)
        self.XGrid = np.array([X.flatten(), Y.flatten()], dtype=np.float).T
        self.idxs = self.tri.find_simplex(self.XGrid)
        self.bary = get_barycentric(self.XGrid, self.idxs, self.tri, self.XKeyWBbox)
        self.colors = self.img[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1, :]/255.0


    def get_face_keypts(self, add_forehead = True, add_neck = True):
        """
        Return the keypoints of the first face detected in the image
        Parameters
        ----------
        img: ndarray(M, N, 3)
            An RGB image which contains at least one face
        Returns
        -------
        XKey: ndarray(71, 2)
            Locations of the facial landmarks.  Last 4 are 4 corners
            of the expanded bounding box
        add_forehead: boolean
            Whether to add 3 points on the forehead based on some
            bounding landmarks on the face
        add_neck: boolean
            Whether to add 2 points on the neck based on some bounding
            landmarks around the jaw
        """

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(self.img, 1)
        d = dets[0]
        # Get the landmarks/parts for the face in box d.
        shape = predictor(self.img, d)
        XKey = shape_to_np(shape)
        # Add three landmarks for the forehead
        if add_forehead:
            x1 = XKey[0, :]
            x2 = XKey[16, :]
            m = x2 - x1
            n = np.array([m[1], -m[0]])
            x1 = x1 + 0.3*n
            x2 = x2 + 0.3*n
            x3 = 0.5*(x1 + x2) + 0.1*n
            XKey = np.concatenate((XKey, x1[None, :], x2[None, :], x3[None, :]), axis=0)
        # Add two landmarks on the neck
        if add_neck:
            x1 = XKey[5, :]
            x2 = XKey[11, :]
            n = np.array([-m[1], m[0]])
            x1 = x1 + 0.3*n
            x2 = x2 + 0.3*n
            XKey = np.concatenate((XKey, x1[None, :], x2[None, :]), axis=0)
        self.XKey = XKey
        return self.XKey
    
    def get_blocks(self, block_size):
        """
        Return indices of the upper left corners of blocks
        that overlap the convex hull of the facial landmarks
        Parameters
        ----------
        block_size: int
            The dimension of the square block
        Returns
        -------
        X: ndarray(M, 2)
            The row and column of the upper left corner of each block
            that overlaps the convex hull of the facial landmarks on
            at least one pixel
        """
        ## Step 1: Setup indices in blocks
        img = self.img
        Is = np.arange(0, img.shape[0], block_size)
        Js = np.arange(0, img.shape[1], block_size)
        Js, Is = np.meshgrid(Js, Is)
        X = np.array([Is.flatten(), Js.flatten()]).T
        Y = get_block_pixel_indices(X, block_size)
        shape = Y.shape
        Y = np.reshape(Y, (shape[0]*shape[1], shape[2]))

        ## Step 2: Compute convex hull of landmarks
        hull = ConvexHull(np.fliplr(self.XKey))
        ns = hull.equations[:, 0:2]
        ps = hull.equations[:, 2].flatten()
        ds = Y.dot(ns.T) + ps[None, :]
        inside = np.sum(ds < 0, 1) == ds.shape[1]
        inside = np.reshape(inside, (shape[0], shape[1]))
        inside = np.sum(inside, 1) > 0
        return X[inside, :]
    
    def get_forward_map(self, XKey2):
        """
        Extend the map from they keypoints to these new keypoints to a refined piecewise
        affine map from triangles to triangles
        Parameters
        ----------
        XKey2: ndarray(71, 2)
            New locations of facial landmarks
        
        Returns
        -------
        imgwarped: ndarray(M, N, 3)
            An image warped according to the map
        """
        [i1, i2, j1, j2] = self.bbox
        XKey2WBbox = add_bbox_to_keypoints(XKey2, self.bbox)
        XGrid2 = barycentric_to_euclidean(self.idxs, self.tri, XKey2WBbox, self.bary)
        diff = XGrid2 - self.XGrid
        XGrid2 = self.XGrid - diff
        XGrid2 = np.fliplr(XGrid2)
        # Numerical precision could cause coords to be out of bounds
        XGrid2[XGrid2[:, 0] <= np.min(self.pixy), 0] = np.min(self.pixy)
        XGrid2[XGrid2[:, 0] >= np.max(self.pixy), 0] = np.max(self.pixy)
        XGrid2[XGrid2[:, 1] <= np.min(self.pixx), 1] = np.min(self.pixx)
        XGrid2[XGrid2[:, 1] >= np.max(self.pixx), 1] = np.max(self.pixx)
        imgret = np.array(self.img)
        shape = (i2-i1+1, j2-j1+1)
        for c in range(3):
            interpbox = interpn((self.pixy, self.pixx), self.colors[:, :, c], XGrid2)
            interpbox = np.array(np.round(255*interpbox), dtype = np.uint8)
            imgret[i1:i2+1, j1:j2+1, c] = np.reshape(interpbox, shape)
        return imgret

    def get_good_points_to_track(self):
        import cv2
        mask = get_mask(self.XKey, self.img.shape)
        for region_idx in [LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, NOSE_LANDMARKS, MOUTH_LANDMARKS]:
            region = self.XKey[region_idx, :]
            maskr = get_mask(region, self.img.shape, fuzz=5)
            mask = mask^maskr
        mask[mask < 0] = 0
        mask = np.array(mask, dtype=np.uint8)
        
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        feature_params = dict( maxCorners = 1000,
                            qualityLevel = 0.01,
                            minDistance = 10,
                            blockSize = 7,
                            gradientSize = 7 )
        p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        return p0



    def plotKeypoints(self, drawLandmarks = True, numberLandmarks = False, drawTriangles = False):
        """
        Plot the image with the keypoints superimposed
        """
        plt.imshow(self.img)
        if drawLandmarks:
            plt.scatter(self.XKey[:, 0], self.XKey[:, 1])
        if numberLandmarks:
            for i in range(self.XKey.shape[0]):
                plt.text(self.XKey[i, 0], self.XKey[i, 1], "{}".format(i))
        if drawTriangles:
            plt.triplot(self.XKeyWBbox[:, 0], self.XKeyWBbox[:, 1], self.tri.simplices)


def test_flow():
    filename = "CVPR2014Data/1/00001.jpg"
    face = MorphableFace(filename)
    face.get_face_keypts()
    face.plotKeypoints(numberLandmarks=True)
    plt.show()
    bbox = face.get_bbox()
    p0 = face.get_good_points_to_track()
    plt.imshow(face.img)
    plt.scatter(p0[:, 0, 0], p0[:, 0, 1])
    plt.xlim(bbox[2], bbox[3])
    plt.ylim(bbox[1], bbox[0])
    plt.show()

def test_delaunay(filename):
    face = MorphableFace(filename)
    face.get_face_keypts()
    face.setup_grid(face.get_bbox())
    plt.subplot(221)
    face.plotKeypoints()
    plt.title("DLib Facial Landmarks")
    plt.subplot(222)
    face.plotKeypoints(False, True)
    plt.title("Delaunay Triangulation")
    plt.subplot(223)
    face.plotKeypoints()
    block_size = 25
    X = face.get_blocks(block_size)
    X += int(block_size/2)
    plt.scatter(X[:, 1], X[:, 0], 1)
    plt.title("{} {}x{} Blocks".format(X.shape[0], block_size, block_size))
    plt.tight_layout()
    plt.show()

def test_warp(filename):
    """
    Make sure the warping is working by randomly perturbing the
    facial landmarks a bunch of times
    """
    face = MorphableFace(filename)
    face.get_face_keypts()
    bbox = face.get_bbox()
    expand_bbox(bbox, 0.2, face.img.shape)
    face.setup_grid(bbox)
    NFrames = 10
    for f in range(NFrames):
        plt.clf()
        print("Warping frame %i of %i..."%(f+1, NFrames))
        XKey2 = np.array(face.XKey, dtype=float)
        XKey2 += 2*np.random.randn(XKey2.shape[0], 2)
        tic = time.time()
        res = face.get_forward_map(XKey2)
        plt.imshow(res)
        print("Elapsed Time: %.3g"%(time.time()-tic))
        plt.scatter(XKey2[:, 0], XKey2[:, 1], 2)
        plt.savefig("WarpTest%i.png"%f)

if __name__ == '__main__':
    filename = "CVPR2014Data/1/00001.jpg"
    #test_delaunay(filename)
    test_flow()