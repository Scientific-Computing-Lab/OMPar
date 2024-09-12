{
  Real max_cache[BLOCK_SIZE];
  {
    int lid = omp_get_thread_num();
    int tid = omp_get_team_num();
    int gid = tid * BLOCK_SIZE + lid;
    int row = (gid % (NUM/2)) + 1; 
    int col = (gid / (NUM/2)) + 1; 

    // allocate shared memory to store maximum velocities
    max_cache[lid] = ZERO;

    int NUM_2 = NUM >> 1;
    Real new_v = ZERO;

    if (row != NUM_2) {
      Real p_ij, p_ijp1, new_v2;

      // red pressure point
      p_ij = pres_red(col, row);
      p_ijp1 = pres_black(col, row + ((col + 1) & 1));

      new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
      v(col, (2 * row) - (col & 1)) = new_v;


      // black pressure point
      p_ij = pres_black(col, row);
      p_ijp1 = pres_red(col, row + (col & 1));

      new_v2 = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
      v(col, (2 * row) - ((col + 1) & 1)) = new_v2;


      // check for max of these two
      new_v = fmax(fabs(new_v), fabs(new_v2));

      if (col == NUM) {
        // also test for max velocity at vertical boundary
        new_v = fmax(new_v, fabs( v(NUM + 1, (2 * row)) ));
      }

    } else {

      if ((col & 1) == 1) {
        // black point is on boundary, only calculate red point below it
        Real p_ij = pres_red(col, row);
        Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

        new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
        v(col, (2 * row) - (col & 1)) = new_v;

      } else {
        // red point is on boundary, only calculate black point below it
        Real p_ij = pres_black(col, row);
        Real p_ijp1 = pres_red(col, row + (col & 1));

        new_v = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
        v(col, (2 * row) - ((col + 1) & 1)) = new_v;
      }

      // get maximum v velocity
      new_v = fabs(new_v);

      // check for maximum velocity in boundary cells also
      new_v = fmax(fabs( v(col, NUM) ), new_v);
      new_v = fmax(fabs( v(col, 0) ), new_v);

      new_v = fmax(fabs( v(col, NUM + 1) ), new_v);

    } // end if

    // store absolute value of velocity
    max_cache[lid] = new_v;

    // synchronize threads in block to ensure all velocities stored

    // calculate maximum for block
    int i = BLOCK_SIZE >> 1;
    while (i != 0) {
      if (lid < i) {
        max_cache[lid] = fmax(max_cache[lid], max_cache[lid + i]);
      }
      i >>= 1;
    }

    // store block's summed residuals
    if (lid == 0) {
      max_v_arr[tid] = max_cache[0];
    }
  }
}
