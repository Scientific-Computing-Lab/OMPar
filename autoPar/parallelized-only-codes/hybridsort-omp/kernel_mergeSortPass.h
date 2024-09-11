for (int gid = 0; gid < global[0]; gid++) {
  // The division to work on
  int division = gid / threadsPerDiv;
  if(division < DIVISIONS) {
    // The block within the division
    int int_gid = gid - division * threadsPerDiv;
    int Astart = startaddr[division] + int_gid * nrElems;

    int Bstart = Astart + nrElems/2;
    //global float4 *resStart;
    float4* resStart= &(d_resultList[Astart]);

    if(Astart < startaddr[division + 1]) {
      if(Bstart >= startaddr[division + 1]){
        for(int i=0; i<(startaddr[division + 1] - Astart); i++)
        {
          resStart[i] = d_origList[Astart + i];
        }
      } else {

        int aidx = 0;
        int bidx = 0;
        int outidx = 0;
        float4 a, b;
        float4 zero = {0.f, 0.f, 0.f, 0.f};
        a = d_origList[Astart + aidx];
        b = d_origList[Bstart + bidx];

        while(true)
        {
          /**
           * For some reason, it's faster to do the texture fetches here than
           * after the merge
           */
          float4 nextA = d_origList[Astart + aidx + 1];
          float4 nextB = (Bstart + bidx + 1 >= listsize/4) ? zero : d_origList[Bstart + bidx + 1];

          float4 na = getLowest(a,b);
          float4 nb = getHighest(a,b);
          a = sortElem(na);
          b = sortElem(nb);
          // Now, a contains the lowest four elements, sorted
          resStart[outidx++] = a;

          bool elemsLeftInA;
          bool elemsLeftInB;

          elemsLeftInA = (aidx + 1 < nrElems/2); // Astart + aidx + 1 is allways less than division border
          elemsLeftInB = (bidx + 1 < nrElems/2) && (Bstart + bidx + 1 < startaddr[division + 1]);

          if(elemsLeftInA){
            if(elemsLeftInB){
              float nextA_t = nextA.x;
              float nextB_t = nextB.x;
              if(nextA_t < nextB_t) { aidx += 1; a = nextA; }
              else { bidx += 1;  a = nextB; }
            }
            else {
              aidx += 1; a = nextA;
            }
          }
          else {
            if(elemsLeftInB){
              bidx += 1;  a = nextB;
            }
            else {
              break;
            }
          }
        }
        resStart[outidx++] = b;
      }
    }
  }
}
