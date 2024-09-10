for (int ind = 1; ind < NUM+1; ind++) {

  // left boundary
  u(0, ind) = ZERO;
  v(0, ind) = -v(1, ind);

  // right boundary
  u(NUM, ind) = ZERO;
  v(NUM + 1, ind) = -v(NUM, ind);

  // bottom boundary
  u(ind, 0) = -u(ind, 1);
  v(ind, 0) = ZERO;

  // top boundary
  u(ind, NUM + 1) = TWO - u(ind, NUM);
  v(ind, NUM) = ZERO;

  if (ind == NUM) {
    // left boundary
    u(0, 0) = ZERO;
    v(0, 0) = -v(1, 0);
    u(0, NUM + 1) = ZERO;
    v(0, NUM + 1) = -v(1, NUM + 1);

    // right boundary
    u(NUM, 0) = ZERO;
    v(NUM + 1, 0) = -v(NUM, 0);
    u(NUM, NUM + 1) = ZERO;
    v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

    // bottom boundary
    u(0, 0) = -u(0, 1);
    v(0, 0) = ZERO;
    u(NUM + 1, 0) = -u(NUM + 1, 1);
    v(NUM + 1, 0) = ZERO;

    // top boundary
    u(0, NUM + 1) = TWO - u(0, NUM);
    v(0, NUM) = ZERO;
    u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
    v(ind, NUM + 1) = ZERO;
  } // end if
}
