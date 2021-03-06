import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    return XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)

##########################################################
##       DELAUNAY/PIECEWISE AFFINE UTILITIES            ##
##########################################################

def get_barycentric(X, idxs, tri, XTri):
    """
    Get the barycentric coordinates of all points
    https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    
    Parameters
    ---------- 
    X: ndarray (N, 2)
        Points of which to compute barycentric coordinates
    idxs: ndarray(N)
        Indices of each point into the Delaunay triangulation.
    tri: object
        Delaunay triangulation data structure
    XTri: ndarray (K, 2)
        Landmark points into which the delaunay triangle indexes
    
    Returns
    -------
    bary: ndarray (N, 3)
        The barycentric coordinates of each point with respect to its triangle
    """
    a, b, c = [XTri[tri.simplices[idxs][:, k], :] for k in range(3)]
    v0 = b - a
    v1 = c - a
    v2 = X - a
    d00 = np.sum(v0*v0, 1)
    d01 = np.sum(v0*v1, 1)
    d11 = np.sum(v1*v1, 1)
    d20 = np.sum(v2*v0, 1)
    d21 = np.sum(v2*v1, 1)
    denom = d00*d11 - d01*d01
    v = (d11*d20 - d01*d21)/denom
    w = (d00*d21 - d01*d20)/denom
    u = 1.0 - v - w
    return np.array([u, v, w]).T

def barycentric_to_euclidean(idxs, tri, XTri, bary):
    """
    Return Euclidean coordinates from a barycentric coordinates
    idxs: ndarray(N)
        Indices of each point into the Delaunay triangulation.
    tri: object
        Delaunay triangulation data structure
    XTri: ndarray (K, 2)
        Landmark points into which the delaunay triangle indexes
    bary: ndarray(N, 3)
        Barycentric coordinates of each point in X with respect to its triangle
    """
    a, b, c = [XTri[tri.simplices[idxs][:, k], :] for k in range(3)]
    u, v, w = bary[:, 0], bary[:, 1], bary[:, 2]
    return u[:, None]*a + v[:, None]*b + w[:, None]*c


def test_barycentric_tri():
    """
    Testing point location and barycentric coordinates.
    Plot the indices for the simplices to figure out how point
    location happened, and then go to barycentric and back
    to make sure it's the identity
    """
    np.random.seed(2)
    XTri = np.random.rand(6, 2)
    tri = Delaunay(XTri)
    N = 100
    pix = np.linspace(0, 1, N)
    J, I = np.meshgrid(pix, pix)
    XGrid = np.array([J.flatten(), I.flatten()]).T
    allidxs = tri.find_simplex(XGrid)
    idxs = allidxs[allidxs > -1]
    XGrid = XGrid[allidxs > -1, :]

    bary = get_barycentric(XGrid, idxs, tri, XTri)
    XGridBack = barycentric_to_euclidean(idxs, tri, XTri, bary)

    print("Is Cartesian -> Barycentric -> Cartesian Identity?: %s"%np.allclose(XGrid, XGridBack))

    plt.subplot(121)
    plt.imshow(np.reshape(allidxs, (N, N)), extent=(0, 1, 1, 0))
    plt.triplot(XTri[:, 0], XTri[:, 1], triangles=tri.simplices.copy())
    plt.subplot(122)
    #plt.scatter(XGrid[:, 0], XGrid[:, 1], 20, bary)
    plt.scatter(XGridBack[:, 0], XGridBack[:, 1])
    plt.triplot(XTri[:, 0], XTri[:, 1], triangles=tri.simplices.copy(), c='C1')
    plt.show()

# Below is some code to find containing triangles
# for the case where the triangulation isn't Delaunay
def sign(x, p2, p3):
    return (x[0] - p3[:, 0])*(p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0])*(x[1] - p3[:, 1])

def get_triangle_idx(X, tri, x):
    # Check to see if on all sides
    [a, b, c] = [X[tri[:, k], :] for k in range(3)]
    b1 = sign(x, a, b) < 0
    b2 = sign(x, b, c) < 0
    b3 = sign(x, c, a) < 0
    agree = np.array((b1 == b2), dtype=int) + np.array((b2 == b3), dtype=int)
    return np.arange(agree.size)[agree == 2][0]

def test_tri_idx():
    np.random.seed(0)
    X = np.random.randn(10, 2)
    x = np.random.rand(10, 2)
    tri = Delaunay(X)

    for k in range(x.shape[0]):
        idx = get_triangle_idx(X, tri.simplices, x[k, :])
        print(idx)
    print(tri.find_simplex(x))


##########################################################
##               BINARY MASK UTILITIES                  ##
##########################################################

def get_mask(XLand, shape, fuzz = 0):
    """
    Compute a binary mask image based on the convex hull
    of a set of landmarks
    Parameters
    ----------
    XLand: ndarray(M, 2)
        A set of M landmarks.  x coordinates in first column,
        y coordinates in second column
    shape: (Height, Width)
        Dimensions of the mask
    fuzz: float
        Amount by which to expand mask around convex hull
    """
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    Y = np.array([X.flatten(), Y.flatten()]).T
    hull = ConvexHull(XLand)
    ns = hull.equations[:, 0:2]
    ps = hull.equations[:, 2].flatten()
    ds = Y.dot(ns.T) + ps[None, :]
    inside = np.sum(ds < fuzz, 1) == ds.shape[1]
    inside = np.reshape(inside, (shape[0], shape[1]))
    return inside

def get_block_pixel_indices(X, block_size):
    """
    Given the locations of the upper left of blocks and the block dimensions,
    return an array of row/column indices of all pixels in the blocks
    Parameters
    ----------
    X: ndarray(M, 2)
        Row/column indices of upper left corners of M blocks
    block_size: int
        The dimension of the block
    Returns
    -------
    Y: ndarray(M, block_size^2, 2)
        Index 0: Block, Index 1: pixels in block, Index 2: row/col
    """
    pix = np.arange(block_size)
    pixj, pixi = np.meshgrid(pix, pix)
    block = np.array([pixi.flatten(), pixj.flatten()])
    Y = X[:, :, None] + block[None, :, :]
    Y = np.swapaxes(Y, 1, 2)
    return Y


##########################################################
##              BOUNDING BOX UTILITIES                  ##
##########################################################

def clamp_bbox(bbox, shape):
    """
    Clamp a bounding box to the dimensions of an array
    Parameters
    ----------
    bbox: ndarray([i1, i2, j1, j2])
        A bounding box
    shape: tuple
        Dimensions to which to clamp
    """
    [i1, i2, j1, j2] = bbox
    j1 = max(0, int(j1))
    i1 = max(0, int(i1))
    j2 = min(shape[1]-1, int(j2))
    i2 = min(shape[0]-1, int(i2))
    bbox[0:4] = [i1, i2, j1, j2]
    return bbox

def expand_bbox(bbox, pad, shape):
    """
    Expand a bounding box by a certain factor
    in all directions
    Parameters
    ----------
    bbox: ndarray([i1, i2, j1, j2])
        A bounding box
    pad: float
        The fraction by which to expand (>0 is bigger, <0 is smaller)
    shape: tuple
        Dimensions to which to clamp
    """
    [i1, i2, j1, j2] = bbox
    width = j2-j1+1
    height = i2-i1+1
    i1 -= pad*height
    i2 += pad*height
    j1 -= pad*width
    j2 += pad*width
    bbox[0:4] = [i1, i2, j1, j2]
    return clamp_bbox(bbox, shape)

def bbox_union(bbox1, bbox2):
    """
    Compute the union of two bounding boxes
    Parameters
    ----------
    bbox1: ndarray([i1, i2, j1, j2])
        First bounding box
    bbox2: ndarray([i1, i2, j1, j2])
        Second bounding box
    """
    i1, j1 = np.minimum(bbox1[[0, 2]], bbox2[[0, 2]])
    i2, j2 = np.maximum(bbox1[[1, 3]], bbox2[[1, 3]])
    return np.array([i1, i2, j1, j2])

def add_bbox_to_keypoints(keypoints, bbox):
    """
    Create a new set of keypoints with corners of the
    bounding box
    """
    [i1, i2, j1, j2] = bbox
    corners = np.array([[j1, i1], [j1, i2], [j2, i2], [j2, i1]])
    return np.concatenate((keypoints, corners), axis=0)

if __name__ == '__main__':
    #test_barycentric_tri()
    test_tri_idx()