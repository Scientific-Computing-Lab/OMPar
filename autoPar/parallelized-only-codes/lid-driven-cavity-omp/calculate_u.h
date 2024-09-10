{
  Real max_cache[BLOCK_SIZE];
  {
    int lid = omp_get_thread_num();
    int tid = omp_get_team_num();
    int gid = tid * BLOCK_SIZE + lid;
    int row = (gid % (NUM/2)) + 1; 
    int col = (gid / (NUM/2)) + 1; 

    // allocate shared memory to store max velocities
    max_cache[lid] = ZERO;

    int NUM_2 = NUM >> 1;
    Real new_u = ZERO;

    if (col != NUM) {

      Real p_ij, p_ip1j, new_u2;

      // red point
      p_ij = pres_red(col, row);
      p_ip1j = pres_black(col + 1, row);

      new_u = F(col, (2 * row) - (col & 1)) - (dt * (p_ip1j - p_ij) / dx);
      u(col, (2 * row) - (col & 1)) = new_u;

      // black point
      p_ij = pres_black(col, row);
      p_ip1j = pres_red(col + 1, row);

      new_u2 = F(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ip1j - p_ij) / dx);
      u(col, (2 * row) - ((col + 1) & 1)) = new_u2;

      // check for max of these two
      new_u = fmax(fabs(new_u), fabs(new_u2));

      if ((2 * row) == NUM) {
        // also test for max velocity at vertical boundary
        new_u = fmax(new_u, fabs( u(col, NUM + 1) ));
      }
    } else {
      // check for maximum velocity in boundary cells also
      new_u = fmax(fabs( u(NUM, (2 * row)) ), fabs( u(0, (2 * row)) ));
      new_u = fmax(fabs( u(NUM, (2 * row) - 1) ), new_u);
      new_u = fmax(fabs( u(0, (2 * row) - 1) ), new_u);

      new_u = fmax(fabs( u(NUM + 1, (2 * row)) ), new_u);
      new_u = fmax(fabs( u(NUM + 1, (2 * row) - 1) ), new_u);

    } // end if

    // store maximum u for block from each thread
    max_cache[lid] = new_u;

    // synchronize threads in block to ensure all velocities stored

    // calculate maximum for block
    int i = BLOCK_SIZE >> 1;
    while (i != 0) {
      if (lid < i) {
        max_cache[lid] = fmax(max_cache[lid], max_cache[lid + i]);
      }
      i >>= 1;
    }

    // store block's maximum
    if (lid == 0) {
      max_u_arr[tid] = max_cache[0];
    }
  }
}
