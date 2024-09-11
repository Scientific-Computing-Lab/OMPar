////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute per-workgroup partial histograms
////////////////////////////////////////////////////////////////////////////////
{
  unsigned int s_Hist[HISTOGRAM_BLOCK_MEMORY];
  {

    const int lid = omp_get_thread_num();
    const int lsize = omp_get_num_threads();
    const int tid = omp_get_team_num();
    const int gid = tid * lsize + lid; 
    const int gsize = omp_get_num_teams() * lsize;

    //Per-warp substorage storage
    int mulBase = (lid >> BUCKET_WARP_LOG_SIZE);
    const int warpBase = IMUL(mulBase, HISTOGRAM_BIN_COUNT);

    //Clear shared memory storage for current threadblock before processing
    for(uint i = lid; i < HISTOGRAM_BLOCK_MEMORY; i+=lsize) {
      s_Hist[i] = 0;
    }


    //Read through the entire input buffer, build per-warp histograms
    for(int pos = gid; pos < listsize; pos += gsize) {
      uint data4 = ((d_input[pos] - minimum)/(maximum - minimum)) * HISTOGRAM_BIN_COUNT;
      s_Hist[warpBase+(data4 & 0x3FFU)]++;
    }

    //Per-block histogram reduction

    for(int pos = lid; pos < HISTOGRAM_BIN_COUNT; pos += lsize){
      uint sum = 0;
      for(int i = 0; i < HISTOGRAM_BLOCK_MEMORY; i+= HISTOGRAM_BIN_COUNT){ 
        sum += s_Hist[pos + i] & 0x07FFFFFFU;
      }
      h_offsets[pos] += sum;
    }
  }
}
