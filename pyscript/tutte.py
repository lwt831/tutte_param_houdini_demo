import hou
import utils
import numpy as np
from importlib import reload
import laplacian_matrix
import find_boundary_loop
import tutte_cpp
from profile_tool import Profile

reload(laplacian_matrix)
reload(find_boundary_loop)
from find_boundary_loop import FindBoundaryLoop
from laplacian_matrix import ComputeCotLaplacianMatrix
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve


@Profile(follow=[])
def Parameterization(geo, use_cpp):
    # nf = utils.GetNumberOfFaces(geo)
    npts = utils.GetNumberOfPoints(geo)
    points = utils.GetPoints(geo)
    faces = utils.GetFaces(geo)
    if use_cpp:
        print("use cpp")
        uv = tutte_cpp.Parameterization(points, faces)
    else:
        print("use python")
        boundary_loop = FindBoundaryLoop(faces)
        boundary_loop = boundary_loop[::-1]
        laplacian_matrix = ComputeCotLaplacianMatrix(points, faces)

        # border condition
        eye_b = eye(npts).tocsr()[boundary_loop]
        nb = boundary_loop.shape[0]
        laplacian_matrix[boundary_loop] = eye_b
        # solve laplacian_matrix * x = rhs
        rhs = np.zeros((npts, 2))
        angles = np.linspace(0, 2 * np.pi, nb, endpoint=False)
        circle = np.column_stack((np.cos(angles), np.sin(angles))) * 0.5 + np.array(
            [0.5, 0.5]
        )
        # print(boundary_loop)
        rhs[boundary_loop] = circle
        uv = spsolve(laplacian_matrix, rhs)
    utils.AddPointsAttrib(geo, "uv", (0.0, 0.0))
    utils.SetPointsFloatAttrib(geo, "uv", uv.reshape(-1))
