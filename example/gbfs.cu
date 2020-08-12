#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/bfs.hpp"
#include "test/test.hpp"

#include "graphblas/io/data_loader.h"

using namespace graphblas::io;


bool debug_;
bool memory_;

int main(int argc, char** argv) {
  // Read in sparse matrix
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " [scipy-npz-filename]" << std::endl;
    exit(EXIT_FAILURE);
  }
  CSRMatrix<float> csr_matrix_float = load_csr_matrix_from_float_npz(std::string(argv[argc-1]));
  graphblas::Index nrows = csr_matrix_float.num_rows;
  graphblas::Index ncols = csr_matrix_float.num_cols;
  std::vector<float> vals = csr_matrix_float.adj_data;
  graphblas::Index nvals = vals.size();
  std::vector<graphblas::Index> col_indices;
  std::copy(csr_matrix_float.adj_indices.begin(), csr_matrix_float.adj_indices.end(),
            std::back_inserter(col_indices));
  std::vector<graphblas::Index> row_indices(nvals);
  for (size_t row_idx = 0; row_idx < nrows; row_idx++) {
    size_t start = csr_matrix_float.adj_indptr[row_idx];
    size_t end = csr_matrix_float.adj_indptr[row_idx+1];
    for (size_t i = start; i < end; i++) {
      row_indices[i] = row_idx;
    }
  }

  // Arguments
  int directed = 1;
  int num_runs = 5;
  int source = 0;

  // Descriptor
  graphblas::Descriptor desc;
  po::variables_map vm;
  parseArgs(argc, argv, &vm);
  CHECK(desc.loadArgs(vm));

  // Matrix A
  graphblas::Matrix<float> a(nrows, ncols);
  a.build(&row_indices, &col_indices, &vals, nvals, GrB_NULL, NULL);

  // Vector v
  graphblas::Vector<float> v(nrows);

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::bfs(&v, &a, source, &desc);
  warmup.Stop();

  // Benchmark
  graphblas::Vector<float> y(nrows);
  CpuTimer vxm_gpu;
  vxm_gpu.Start();
  float tight = 0.f;
  float val;
  for (int i = 0; i < num_runs; i++) {
    val = graphblas::algorithm::bfs(&y, &a, source, &desc);
    tight += val;
  }
  vxm_gpu.Stop();

  std::cout << "warmup: " << warmup.ElapsedMillis() << " ms" <<std::endl;
  float elapsed_vxm = vxm_gpu.ElapsedMillis();
  // std::cout << "tight: " << tight / num_runs << " ms" <<std::endl;
  std::cout << "run: " << elapsed_vxm / num_runs << " ms" <<std::endl;

  return 0;
}
