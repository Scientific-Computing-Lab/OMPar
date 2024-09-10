for (int col = 1; col < NUM+1; col++)
{
  for (int row = 1; row < NUM/2+1; row++)
  {

    int NUM_2 = NUM >> 1;

    Real p_ij = pres_black(col, row);

    Real p_im1j = pres_red(col - 1, row);
    Real p_ip1j = pres_red(col + 1, row);
    Real p_ijm1 = pres_red(col, row - ((col + 1) & 1));
    Real p_ijp1 = pres_red(col, row + (col & 1));

    // right-hand side
    Real rhs = (((F(col, (2 * row) - ((col + 1) & 1))
            - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
        + ((G(col, (2 * row) - ((col + 1) & 1))
            - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

    pres_black(col, row) = p_ij * (ONE - omega) + omega * 
      (((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
       rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));
  }
}
