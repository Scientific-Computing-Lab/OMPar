{
  Real sum_cache[BLOCK_SIZE];
  {

    int lid = omp_get_thread_num();
    int tid = omp_get_team_num();
    int gid = tid * BLOCK_SIZE + lid;
    int row = (gid % (NUM/2)) + 1; 
    int col = (gid / (NUM/2)) + 1; 

    int NUM_2 = NUM >> 1;

    Real p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs, res, res2;

    // red point
    p_ij = pres_red(col, row);

    p_im1j = pres_black(col - 1, row);
    p_ip1j = pres_black(col + 1, row);
    p_ijm1 = pres_black(col, row - (col & 1));
    p_ijp1 = pres_black(col, row + ((col + 1) & 1));

    rhs = (((F(col, (2 * row) - (col & 1)) - F(col - 1, (2 * row) - (col & 1))) / dx)
        +  ((G(col, (2 * row) - (col & 1)) - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

    // calculate residual
    res = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
      + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

    // black point
    p_ij = pres_black(col, row);

    p_im1j = pres_red(col - 1, row);
    p_ip1j = pres_red(col + 1, row);
    p_ijm1 = pres_red(col, row - ((col + 1) & 1));
    p_ijp1 = pres_red(col, row + (col & 1));

    // right-hand side
    rhs = (((F(col, (2 * row) - ((col + 1) & 1)) - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
        +  ((G(col, (2 * row) - ((col + 1) & 1)) - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

    // calculate residual
    res2 = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
      + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

    sum_cache[lid] = (res * res) + (res2 * res2);

    // synchronize threads in block to ensure all residuals stored

    // add up squared residuals for block
    int i = BLOCK_SIZE >> 1;
    while (i != 0) {
      if (lid < i) {
        sum_cache[lid] += sum_cache[lid + i];
      }
      i >>= 1;
    }

    // store block's summed residuals
    if (lid == 0) {
      res_arr[tid] = sum_cache[0];
    }
  }
}

