for (int col = 1; col < NUM+1; col++)
{
  for (int row = 1; row < NUM+1; row++)
  {
    if (col == NUM) {
      // right boundary, F_ij = u_ij
      // also do left boundary
      F(0, row) = u(0, row);
      F(NUM, row) = u(NUM, row);
    } else {

      // u velocities
      Real u_ij = u(col, row);
      Real u_ip1j = u(col + 1, row);
      Real u_ijp1 = u(col, row + 1);
      Real u_im1j = u(col - 1, row);
      Real u_ijm1 = u(col, row - 1);

      // v velocities
      Real v_ij = v(col, row);
      Real v_ip1j = v(col + 1, row);
      Real v_ijm1 = v(col, row - 1);
      Real v_ip1jm1 = v(col + 1, row - 1);

      // finite differences
      Real du2dx, duvdy, d2udx2, d2udy2;

      du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
          + mix_param * (fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
            - fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (FOUR * dx);
      duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
          + mix_param * (fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
            - fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (FOUR * dy);
      d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) / (dx * dx);
      d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) / (dy * dy);

      F(col, row) = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);

    } // end if
  }
}
