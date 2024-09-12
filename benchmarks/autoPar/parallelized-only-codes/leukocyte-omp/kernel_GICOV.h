
for (int gid = 0; gid < global_work_size; gid++) {

  // Determine this thread's pixel
  int i = gid/local_work_size + MAX_RAD + 2;
  int j = gid%local_work_size + MAX_RAD + 2;

  // Initialize the maximal GICOV score to 0
  float max_GICOV = 0.f;

  // Iterate across each stencil
  for (int k = 0; k < NCIRCLES; k++) {
    // Variables used to compute the mean and variance
    //  of the gradients along the current stencil
    float sum = 0.f, M2 = 0.f, mean = 0.f;    

    // Iterate across each sample point in the current stencil
    for (int n = 0; n < NPOINTS; n++) {
      // Determine the x- and y-coordinates of the current sample point
      int y = j + host_tY[(k * NPOINTS) + n];
      int x = i + host_tX[(k * NPOINTS) + n];

      // Compute the combined gradient value at the current sample point
      int addr = x * grad_m + y;
      float p = host_grad_x[addr] * host_cos_angle[n] + 
        host_grad_y[addr] * host_sin_angle[n];

      // Update the running total
      sum += p;

      // Partially compute the variance
      float delta = p - mean;
      mean = mean + (delta / (float) (n + 1));
      M2 = M2 + (delta * (p - mean));
    }

    // Finish computing the mean
    mean = sum / ((float) NPOINTS);

    // Finish computing the variance
    float var = M2 / ((float) (NPOINTS - 1));

    // Keep track of the maximal GICOV value seen so far
    if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
  }

  // Store the maximal GICOV value
  host_gicov[(i * grad_m) + j] = max_GICOV;
}
