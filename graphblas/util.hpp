#include <vector>
#include <iostream>
#include <typeinfo>
#include <cstdio>

#include <graphblas/types.hpp>

template<typename T, typename mtxT>
void readTuples( std::vector<graphblas::Index>& row_indices,
			     std::vector<graphblas::Index>& col_indices,
			     std::vector<T>& values,
			     const graphblas::Index nvals,
                 FILE* f)
{
  bool is_weighted = true;
  int c;
  graphblas::Index row_ind, col_ind;
  T value;
  mtxT raw_value;
  char type_str[3];
  type_str[0] = '%';
  if( typeid(mtxT)==typeid(int) )
    type_str[1] = 'd';
  else if( typeid(mtxT)==typeid(float) )
    type_str[1] = 'f';

  int csr_max = 0;
  int csr_current = 0;
  int csr_row = 0;
  int csr_first = 0;

  // Currently checks if there are fewer rows than promised
  // Could add check for edges in diagonal of adjacency matrix
  for( graphblas::Index i=0; i<nvals; i++ ) {
    if( fscanf(f, "%d", &row_ind)==EOF ) {
      std::cout << "Error: not enough rows in mtx file.\n";
      return;
    } else {
      fscanf(f, "%d", &col_ind);

      // Convert 1-based indexing MTX to 0-based indexing C++
      row_indices.push_back(row_ind-1);
      col_indices.push_back(col_ind-1);
      values.push_back(value);

      mtxT raw_value;
      fscanf(f, type_str, &raw_value);
      value = (T) raw_value;
      //std::cout << "The first row is " << row_ind-1 << " " <<  col_ind-1 << std::endl;

      // Finds max csr row.
      if( i!=0 ) {
        if( col_ind-1==0 ) csr_first++;
        if( col_ind-1==col_indices[i-1] )
          csr_current++;
        else {
          csr_current++;
          if( csr_current > csr_max ) {
            csr_max = csr_current;
            csr_current = 0;
            csr_row = row_indices[i-1];
          } else
            csr_current = 0;
        }
      }
  }}
  std::cout << "The biggest row was " << csr_row << " with " << csr_max << " elements.\n";
  std::cout << "The first row has " << csr_first << " elements.\n";
}

template<typename T>
void readTuples( std::vector<graphblas::Index>& row_indices,
			     std::vector<graphblas::Index>& col_indices,
			     std::vector<T>& values,
			     const graphblas::Index nvals,
                 FILE* f)
{
  bool is_weighted = true;
  int c;
  graphblas::Index row_ind, col_ind;
  T value;
  int raw_value;

  int csr_max = 0;
  int csr_current = 0;
  int csr_row = 0;
  int csr_first = 0;

  // Currently checks if there are fewer rows than promised
  // Could add check for edges in diagonal of adjacency matrix
  for( graphblas::Index i=0; i<nvals; i++ ) {
    if( fscanf(f, "%d", &row_ind)==EOF ) {
      std::cout << "Error: not enough rows in mtx file.\n";
      return;
    } else {
      fscanf(f, "%d", &col_ind);

      // Convert 1-based indexing MTX to 0-based indexing C++
      row_indices.push_back(row_ind-1);
      col_indices.push_back(col_ind-1);
      values.push_back(value);

      //std::cout << "The first row is " << row_ind-1 << " " <<  col_ind-1 << std::endl;

      // Finds max csr row.
      if( i!=0 ) {
        if( col_ind-1==0 ) csr_first++;
        if( col_ind-1==col_indices[i-1] )
          csr_current++;
        else {
          csr_current++;
          if( csr_current > csr_max ) {
            csr_max = csr_current;
            csr_current = 0;
            csr_row = row_indices[i-1];
          } else
            csr_current = 0;
        }
      }
  }}
  std::cout << "The biggest row was " << csr_row << " with " << csr_max << " elements.\n";
  std::cout << "The first row has " << csr_first << " elements.\n";
}

template<typename T>
void makeSymmetric( std::vector<graphblas::Index> row_indices, 
                    std::vector<graphblas::Index> col_indices,
                    std::vector<T> values, 
					graphblas::Index& nvals,
                    bool remove_self_loops=true ) {

  graphblas::Index shift = 0;
  std::vector<graphblas::Index> indices;

  for( graphblas::Index i=0; i<nvals; i++ ) {
    if( col_indices[i] != row_indices[i] ) {
      row_indices[nvals+i-shift] = col_indices[i];
      col_indices[nvals+i-shift] = row_indices[i];
      values[nvals+i-shift] = values[i];
	  indices.push_back(i);
    } else shift++;
  }
  //print_array(row_indices);
  //print_array(col_indices);

  nvals = 2*nvals-shift;

  // Sort
  
  struct arrayset<T> work = { row_indices, col_indices, values };
  custom_sort(&work, nvals);
  //print_array(row_indices);
  //print_array(col_indices);

  graphblas::Index curr = col_indices[0];
  graphblas::Index last;
  graphblas::Index curr_row = row_indices[0];
  graphblas::Index last_row;

  for( graphblas::Index i=1; i<nvals; i++ ) {
    last = curr;
    last_row = curr_row;
    curr = col_indices[i];
    curr_row = row_indices[i];

    // Self-loops (TODO: make self-loops contingent on whether we 
    // are doing graph algorithm or matrix multiplication)
    if( remove_self_loops && curr_row == curr )
      col_indices[i] = -1;

	// Duplicates
    if( curr == last && curr_row == last_row ) {
      //printf("Curr: %d, Last: %d, Curr_row: %d, Last_row: %d\n", curr, last, 
	  //  curr_row, last_row );
      col_indices[i] = -1;
  }}

  shift=0;

  // Remove self-loops and duplicates marked -1.
  graphblas::Index back = 0;
  for( graphblas::Index i=0; i+shift<nvals; i++ ) {
    if(col_indices[i] == -1) {
      for( shift; back<=nvals; shift++ ) {
        back = i+shift;
        if( col_indices[back] != -1 ) {
          //printf("Swapping %d with %d\n", i, back ); 
          col_indices[i] = row_indices[back];
          row_indices[i] = col_indices[back];
          col_indices[back] = -1;
          break;
  }}}}

  nvals = nvals-shift;
  row_indices.resize(nvals);
  col_indices.resize(nvals);
  values.resize(nvals);
}

template<typename T>
int readMtx( const char *fname,
		     std::vector<graphblas::Index>& row_indices,
	         std::vector<graphblas::Index>& col_indices,
	         std::vector<T>& values )
{
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  graphblas::Index nrows, ncols, nvals;

  if ((f = fopen(fname, "r")) == NULL) {
    printf( "File %s not found", fname );
    exit(1);
  }

  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Read MTX Size
  if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nvals)) !=0)
    exit(1);

  if (mm_is_integer(matcode))
    readTuples<T, int>( row_indices, col_indices, values, nvals, f );
  else if (mm_is_real(matcode))
    readTuples<T, float>( row_indices, col_indices, values, nvals, f );
  else if (mm_is_pattern(matcode))
    readTuples<T>( row_indices, col_indices, values, nvals, f );

  // If graph is symmetric, replicate it out in memory
  if( mm_is_symmetric(matcode) )
    makeSymmetric<T>( row_indices, col_indices, values, nvals, f );

  mm_write_banner(stdout, matcode);
  mm_write_mtx_crd_size(stdout, nrows, ncols, nvals);
}