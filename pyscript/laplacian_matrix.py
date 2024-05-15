import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import norm
import igl


def MeshFaceEdgeLength(points, faces):
    v0 = points[faces[:, 0], :]
    v1 = points[faces[:, 1], :]
    v2 = points[faces[:, 2], :]

    edge0 = np.linalg.norm(v2 - v1, axis=1)
    edge1 = np.linalg.norm(v0 - v2, axis=1)
    edge2 = np.linalg.norm(v1 - v0, axis=1)
    return edge0, edge1, edge2


def MeshCotEntryFromFaceEdgeLength(points, faces):
    edge0, edge1, edge2 = MeshFaceEdgeLength(points, faces)
    cos0 = (edge1**2 + edge2**2 - edge0**2) / (2 * edge1 * edge2)
    cos1 = (edge0**2 + edge2**2 - edge1**2) / (2 * edge0 * edge2)
    cos2 = (edge0**2 + edge1**2 - edge2**2) / (2 * edge0 * edge1)

    # clip value of cos in [-1, 1]
    cos0 = np.clip(cos0, -1, 1)
    cos1 = np.clip(cos1, -1, 1)
    cos2 = np.clip(cos2, -1, 1)

    sin0 = np.sqrt(1 - cos0**2)
    sin1 = np.sqrt(1 - cos1**2)
    sin2 = np.sqrt(1 - cos2**2)
    cot0 = cos0 / sin0
    cot1 = cos1 / sin1
    cot2 = cos2 / sin2
    return cot0, cot1, cot2


def ComputeCotLaplacianMatrix(points, faces):
    cot0, cot1, cot2 = MeshCotEntryFromFaceEdgeLength(points, faces)
    triplet_i = faces[:, np.array([1, 2, 0, 2, 0, 1])]
    triplet_j = faces[:, np.array([2, 0, 1, 1, 2, 0])]
    triplet_v = np.vstack((cot0, cot1, cot2, cot0, cot1, cot2)).T

    triplet_i = triplet_i.reshape(-1)
    triplet_j = triplet_j.reshape(-1)
    triplet_v = triplet_v.reshape(-1)

    npts = points.shape[0]
    laplacian_matrix = -coo_matrix(
        (triplet_v, (triplet_i, triplet_j)), shape=(npts, npts)
    ).tocsr()

    diagonal_entries = np.array(laplacian_matrix.sum(axis=1)).flatten()
    diagonal_matrix = coo_matrix(
        (diagonal_entries, (np.arange(npts), np.arange(npts))), shape=(npts, npts)
    ).tocsr()
    laplacian_matrix = laplacian_matrix - diagonal_matrix
    ## let's compare with libigl
    # l = igl.cotmatrix(points, faces)
    # print("diff = {}".format(norm(laplacian_matrix + 2 * l, "fro")))
    return laplacian_matrix
