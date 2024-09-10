unsigned int LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}

void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

void FSMKernel(
  const int length,
  const unsigned short *__restrict data,
  int *__restrict best,
  unsigned int *__restrict rndstate,
  unsigned char *__restrict bfsm,
  unsigned char *__restrict same,
  int *__restrict smax,
  int *__restrict sbest,
  int *__restrict oldmax)
{
  {
    unsigned char next[FSMSIZE * 2 * POPSIZE];
    {
      int i, d, pc, s, bit, id, misses, rnd;
      unsigned long long myresult, current;
      unsigned char *fsm, state[TABSIZE];

      int lid = omp_get_thread_num();
      int bid = omp_get_team_num();
      fsm = &next[lid * (FSMSIZE * 2)];

      if (lid == 0) {
        oldmax[bid] = 0;
        same[bid] = 0;
      }

      id = lid + bid * POPSIZE;
      rndstate[id] = SEED ^ id;
      LCG_random_init(&rndstate[id]);

      // initial population
      for (i = 0; i < FSMSIZE * 2; i++) {
        fsm[i] = LCG_random(rndstate+id) & (FSMSIZE - 1);
      }

      // run generations until cutoff times no improvement
      do {
        // reset miss counter and initial state
        for (i = 0; i < TABSIZE; i++) state[i] = 0;
        misses = 0;

        // evaluate FSM
        for (i = 0; i < length; i++) {
          d = (int)data[i];
          pc = (d >> 1) & (TABSIZE - 1);
          bit = d & 1;
          s = (int)state[pc];
          misses += bit ^ (s & 1);
          state[pc] = fsm[s + s + bit];
        }
        for (; i < length; i++) {
          d = (int)data[i];
          pc = (d >> 1) & (TABSIZE - 1);
          bit = d & 1;
          s = (int)state[pc];
          misses += bit ^ (s & 1);
          state[pc] = fsm[s + s + bit];
        }

        // determine best FSM
        if (lid == 0) {
          best[2]++;  // increment generation count
          smax[bid] = 0;
          sbest[bid] = 0;
        }
        
        {
         if (smax[bid] < length - misses) smax[bid] = length - misses;
        }
        //#pragma omp barrier

        if (length - misses == smax[bid]) {
          {
            if (sbest[bid] < lid) sbest[bid] = lid;
          }
        }
        bit = 0;
        if (sbest[bid] == lid) {
          // check if there was an improvement
          same[bid]++;
          if (oldmax[bid] < smax[bid]) {
            oldmax[bid] = smax[bid];
            same[bid] = 0;
          }
        } else {
          // select 1/8 of threads for mutation (best FSM does crossover)
          if ((LCG_random(rndstate+id) & 7) == 0) bit = 1;
        }

        if (bit) {
          // mutate best FSM by flipping random bits with 1/4th probability
          for (i = 0; i < FSMSIZE * 2; i++) {
            rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
            fsm[i] = (next[i + sbest[bid] * FSMSIZE * 2] ^ rnd) & (FSMSIZE - 1);
          }
        } else {
          // crossover best FSM with random FSMs using 3/4 of bits from best FSM
          for (i = 0; i < FSMSIZE * 2; i++) {
            rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
            fsm[i] = (fsm[i] & rnd) | (next[i + sbest[bid] * FSMSIZE * 2] & ~rnd);
          }
        }
      } while (same[bid] < CUTOFF);  // end of loop over generations

      // record best result of this block
      if (sbest[bid] == lid) {
        id = bid;
        myresult = length - misses;
        myresult = (myresult << 32) + id;
        current = *((unsigned long long *)best);
        while (myresult > current) {
          //atomicCAS((unsigned long long *)best, current, myresult);
          {
            unsigned long long old = *((unsigned long long *)best);
            *((unsigned long long *)best) = (old == current) ? myresult : old;
          }
          current = *((unsigned long long *)best);
        }
        for (i = 0; i < FSMSIZE * 2; i++) {
          bfsm[id * (FSMSIZE*2) + i] = fsm[i];
        }
      }
    }
  }
}

void MaxKernel(
  int *__restrict best, 
  const unsigned char *__restrict bfsm)
{
  // copy best FSM state assignment over
  {
    int id = best[0];
    for (int i = 0; i < FSMSIZE * 2; i++) {
      best[i + 3] = bfsm[id * (FSMSIZE*2) + i];
    }
  }
}
