for (int col = 1; col < NUM/2+1; col++)
{
  col = (col * 2) - 1;

  int NUM_2 = NUM >> 1;

  // p_i,0 = p_i,1
  pres_black(col, 0) = pres_red(col, 1);
  pres_red(col + 1, 0) = pres_black(col + 1, 1);

  // p_i,jmax+1 = p_i,jmax
  pres_red(col, NUM_2 + 1) = pres_black(col, NUM_2);
  pres_black(col + 1, NUM_2 + 1) = pres_red(col + 1, NUM_2);
}
