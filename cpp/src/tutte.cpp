#define _USE_MATH_DEFINES
#include <igl/boundary_loop.h>
#include <igl/cotmatrix_entries.h>
#include <math.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/SparseLU>
#include <Eigen/eigen>
#include <Eigen/sparse>
#include <iostream>

Eigen::MatrixXd Parameterization(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces) {
  int nv = vertices.rows();
  int nf = faces.rows();
  std::vector<int> boundary_loop;
  igl::boundary_loop(faces, boundary_loop);
  std::set<int> boundary_set(boundary_loop.begin(), boundary_loop.end());
  // build linear system: L * x = rhs

  // step1: build laplacian_matrix
  Eigen::MatrixXd cot_entries;
  igl::cotmatrix_entries(vertices, faces, cot_entries);
  Eigen::SparseMatrix<double> laplacian_matrix(nv, nv);
  laplacian_matrix.reserve(10 * nv);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(12 * nf);
  for (int fid = 0; fid < nf; fid++) {
    for (int e = 0; e < 3; e++) {
      int source = faces(fid, (e + 1) % 3);
      int dest = faces(fid, (e + 2) % 3);
      if (!boundary_set.count(source)) {
        triplets.emplace_back(source, source, -cot_entries(fid, e));
        triplets.emplace_back(source, dest, cot_entries(fid, e));
      }
      if (!boundary_set.count(dest)) {
        triplets.emplace_back(dest, dest, -cot_entries(fid, e));
        triplets.emplace_back(dest, source, cot_entries(fid, e));
      }
    }
  }
  for (auto boundary_v : boundary_loop) {
    triplets.emplace_back(boundary_v, boundary_v, 1.0);
  }
  laplacian_matrix.setFromTriplets(triplets.begin(), triplets.end());
  // step2: set rhs
  Eigen::MatrixXd rhs(nv, 2);
  rhs.setZero();
  int nb = boundary_loop.size();
  auto boundary_vec = Eigen::Map<Eigen::VectorXi>(boundary_loop.data(), nb);
  Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(nb + 1, 0, M_PI * 2);
  rhs(boundary_vec, 0) = angles.array().cos() * 0.5 + 0.5;
  rhs(boundary_vec, 1) = angles.array().sin() * 0.5 + 0.5;

  // step3: solve
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(laplacian_matrix);

  return solver.solve(rhs);
}

PYBIND11_MODULE(tutte_cpp, m) {
  m.def("Parameterization", &Parameterization, "tutte parameterization");
}