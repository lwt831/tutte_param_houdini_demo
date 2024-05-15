# author: lwt831@mail.ustc.edu.cn

import hou
import inlinecpp
import numpy as np


def GetNumberOfFaces(geo):
    # nf = len(geo.prims()) # very slow

    # use inlinecpp to accelerate
    # you can learn it from HDK
    inlinecpp_func = inlinecpp.createLibrary(
        name="GetNumberOfFaces",
        includes="""#include<GU/GU_Detail.h>""",
        function_sources=[
            """
            int GetNumberOfFaces(const GU_Detail* geo){
                return geo->getNumPrimitives();
            }
            """
        ],
    )
    return inlinecpp_func.GetNumberOfFaces(geo)


def GetNumberOfPoints(geo):
    # nf = len(geo.points()) # very slow
    inlinecpp_func = inlinecpp.createLibrary(
        name="GetNumberOfPoints",
        includes="""#include<GU/GU_Detail.h>""",
        function_sources=[
            """
            int GetNumberOfPoints(const GU_Detail* geo){
                return geo->getNumPoints();
            }
            """
        ],
    )
    return inlinecpp_func.GetNumberOfPoints(geo)


# only support for triangle mesh
def GetFaces(geo):
    nf = GetNumberOfFaces(geo)

    # NumPy arrays are stored by default in row-major order
    faces = np.zeros((nf, 3), int) - 1
    inlinecpp_func = inlinecpp.createLibrary(
        name="GetFaces",
        includes="""#include<GU/GU_Detail.h>""",
        function_sources=[
            """
            void GetFaces(const GU_Detail* geo, int* faces){
                int f_idx = 0;
                const GEO_Primitive* prim;
                GA_FOR_ALL_PRIMITIVES(geo, prim){
                    for(GA_Size i = 0; i < prim->getVertexCount(); i++) {
                        const GA_Offset v_offset = prim->getVertexOffset(i);
                        const GA_Offset p_offset = geo->vertexPoint(v_offset);
                        const int p_index = geo->pointIndex(p_offset);
                        faces[f_idx * 3 + i] = p_index;
                    }
                    f_idx++;
                }
                return;
            }
            """
        ],
    )
    inlinecpp_func.GetFaces(geo, faces.ctypes.data)
    faces = np.flip(faces, 1)  # Houdini will reverse the order of vertices on faces
    return faces


def GetPoints(geo):
    npts = GetNumberOfPoints(geo)
    # points = np.array(geo.pointFloatAttribValues("P").reshape(npts,3)) # slow
    points = (
        np.frombuffer(geo.pointFloatAttribValuesAsString("P"), dtype=np.single)
        .reshape(npts, 3)
        .astype(np.double)
    )
    return points


def SetPoints(geo, points):
    points = points.astype(np.single).reshape(-1)
    # geo.setPointFloatAttribValues("P", points)  # slow
    geo.setPointFloatAttribValuesFromString("P", points)


def GetPointsFloatAttrib(geo, attrib_name):
    attrib = np.frombuffer(
        geo.pointFloatAttribValuesAsString(attrib_name), dtype=np.single
    ).astype(np.double)
    return attrib


def SetPointsFloatAttrib(geo, attrib_name, values):
    values = values.astype(np.single).reshape(-1)
    geo.setPointFloatAttribValuesFromString(attrib_name, values)


def GetFacesFloatAttrib(geo, attrib_name):
    attrib = np.frombuffer(
        geo.primFloatAttribValuesAsString(attrib_name), dtype=np.single
    ).astype(np.double)
    return attrib


def SetFacesFloatAttrib(geo, attrib_name, values):
    values = values.astype(np.single).reshape(-1)
    geo.sePrimFloatAttribValuesFromString(attrib_name, values)


def AddPointsAttrib(geo, attrib_name, default_value):
    geo.addAttrib(hou.attribType.Point, attrib_name, default_value)


def AddFacesAttrib(geo, attrib_name, default_value):
    geo.addAttrib(hou.attribType.Prim, attrib_name, default_value)


def GetFaceIndsInGroup(geo, group_name):
    group = geo.findPrimGroup(group_name)
    prims = group.prims()
    face_inds = np.array([prim.number() for prim in prims])
    return face_inds


def DeleteFaces(geo, face_inds):
    faces = geo.prims()
    delete_faces = np.array(faces)[face_inds]
    geo.deletePrims(delete_faces)


def CreateMeshFromMatrix(geo, points, faces):
    # This code looks very concise and is implemented in vectorization, however, it is slowly.
    # You should accelerate this code using inlinecpp if necessary.
    p_handles = geo.createPoints(points)
    polygons = np.array(p_handles)[faces]
    geo.createPolygons(polygons)


# You can encapsulate more Houdini interfaces in this file.
