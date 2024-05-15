import numpy as np


def FindBoundaryLoop(triangles):
    boundary_edges = {}
    nf = triangles.shape[0]
    for idx in range(nf):
        triangle = triangles[idx, :]
        for i in range(3):
            edge = (
                min(triangle[i], triangle[(i + 1) % 3]),
                max(triangle[i], triangle[(i + 1) % 3]),
            )
            if edge in boundary_edges:
                boundary_edges[edge].append(idx)
            else:
                boundary_edges[edge] = [idx]

    boundary_edges = {
        edge: triangle_indices
        for edge, triangle_indices in boundary_edges.items()
        if len(triangle_indices) == 1
    }

    v_2_adjv = {}
    for e in boundary_edges.keys():
        v0 = e[0]
        v1 = e[1]
        if v0 in v_2_adjv:
            v_2_adjv[v0].append(v1)
        else:
            v_2_adjv[v0] = [v1]
        if v1 in v_2_adjv:
            v_2_adjv[v1].append(v0)
        else:
            v_2_adjv[v1] = [v0]

    v_2_visited_flag = {}
    for v in v_2_adjv.keys():
        v_2_visited_flag[v] = False
    boundary_loop = []
    curr_v = list(v_2_adjv.keys())[0]
    boundary_loop.append(curr_v)
    v_2_visited_flag[curr_v] = True
    while True:
        next = v_2_adjv[curr_v][0]
        if v_2_visited_flag[next]:
            next = v_2_adjv[curr_v][1]
        if v_2_visited_flag[next]:
            break
        curr_v = next
        boundary_loop.append(curr_v)
        v_2_visited_flag[curr_v] = True
    return np.array(boundary_loop)
