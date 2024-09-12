for (int col = 1; col < NUM+1; col++)
{
  for (int row = 1; row < NUM+1; row++)
  {

    if (row == NUM) {
      // top and bottom boundaries
      G(col, 0) = v(col, 0);
      G(col, NUM) = v(col, NUM);

    } else {

      // u velocities
      Real u_ij = u(col, row);
      Real u_ijp1 = u(col, row + 1);
      Real u_im1j = u(col - 1, row);
      Real u_im1jp1 = u(col - 1, row + 1);

      // v velocities
      Real v_ij = v(col, row);
      Real v_ijp1 = v(col, row + 1);
      Real v_ip1j = v(col + 1, row);
      Real v_ijm1 = v(col, row - 1);
      Real v_im1j = v(col - 1, row);

      // finite differences
      Real dv2dy, duvdx, d2vdx2, d2vdy2;

      dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
          + mix_param * (fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
            - fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (FOUR * dy);
      duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
          + mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
            - fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (FOUR * dx);
      d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (dx * dx);
      d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (dy * dy);

      G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);

    } // end if
  }
}
