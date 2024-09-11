{
  unsigned int s_offset[BUCKET_BLOCK_MEMORY];
  {
    const int lid = omp_get_thread_num();
    const int lsize = omp_get_num_threads();
    const int tid = omp_get_team_num();
    const int gid = tid * lsize + lid; 
    const int gsize = omp_get_num_teams() * lsize;

    const int warpBase = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = gsize;

    for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
      s_offset[i] = 0;


    for (int i = gid; i < listsize; i += numThreads) {
      float elem = d_input[i];

      int idx  = DIVISIONS/2 - 1;
      int jump = DIVISIONS/4;
      float piv = pivotPoints[idx]; 

      while(jump >= 1){
        idx = (elem < piv) ? (idx - jump) : (idx + jump);
        piv = pivotPoints[idx];
        jump /= 2;
      }
      idx = (elem < piv) ? idx : (idx + 1);

      int offset;
      offset = s_offset[warpBase+idx]++;

      d_indice[i] = (offset << LOG_DIVISIONS) + idx;
    }

    int prefixBase = tid * BUCKET_BLOCK_MEMORY;

    for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
      d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
  }
}
