for (int row = 1; row < NUM/2+1; row++)
{

  int NUM_2 = NUM >> 1;

  // p_0,j = p_1,j
  pres_black(0, row) = pres_red(1, row);
  pres_red(0, row) = pres_black(1, row);

  // p_imax+1,j = p_imax,j
  pres_black(NUM + 1, row) = pres_red(NUM, row);
  pres_red(NUM + 1, row) = pres_black(NUM, row);
}
