{
  unsigned int s_offset[BUCKET_BLOCK_MEMORY];
  {
    const int lid = omp_get_thread_num();
    const int lsize = omp_get_num_threads();
    const int tid = omp_get_team_num();
    const int gid = tid * lsize + lid; 
    const int gsize = omp_get_num_teams() * lsize;

    int prefixBase = tid * BUCKET_BLOCK_MEMORY;
    const int warpBase = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = gsize;

    for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize){
      s_offset[i] = h_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];
    }


    for (int tid = gid; tid < listsize; tid += numThreads){
      float elem = d_input[tid];
      int id = d_indice[tid];
      d_output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >>  LOG_DIVISIONS)] = elem;
    }
  }
}


