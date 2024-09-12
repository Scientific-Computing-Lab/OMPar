/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
#define ELEMENTARY_LOG2SIZE 11

// Put everything together: batched Fast Walsh Transform CPU front-end
void fwtBatchGPU(float *d_Data, int M, int log2N)
{
    int N = 1 << log2N;
    // save the problem size
    const int sN = N;
    const int sM = M;

    // both versions should pass the final verification
    // 256 is the thread block size
#if 1
    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
      const int stride = N/4;
      for (int m = 0; m < sM; m++) {
        for (int pos = 0; pos < sN/4; pos++) {
          const float *d_Src = d_Data  + m * sN;
          float *d_Dst = d_Data + m * sN;
          int lo = pos & (stride - 1);
          int i0 = ((pos - lo) << 2) + lo;
          int i1 = i0 + stride;
          int i2 = i1 + stride;
          int i3 = i2 + stride;

          float D0 = d_Src[i0];
          float D1 = d_Src[i1];
          float D2 = d_Src[i2];
          float D3 = d_Src[i3];

          float T;
          T = D0;
          D0        = D0 + D2;
          D2        = T - D2;
          T = D1;
          D1        = D1 + D3;
          D3        = T - D3;
          T = D0;
          d_Dst[i0] = D0 + D1;
          d_Dst[i1] = T - D1;
          T = D2;
          d_Dst[i2] = D2 + D3;
          d_Dst[i3] = T - D3;
        }
      }
    }
#else  
    const int teamX = N/(4*256);
    const int teamY = M;
    const int numTeams = teamX * teamY;

    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
      const int stride = N/4;
      {
        {
          const int blockIdx_x = omp_get_team_num() % teamX;
          const int threadIdx_x = omp_get_thread_num();
          const int blockDim_x = 256;
          const int gridDim_x = teamX;
          const int blockIdx_y = omp_get_team_num() / teamX;
          const int pos = blockIdx_x * blockDim_x + threadIdx_x;
          const int   N = blockDim_x * gridDim_x * 4;

          const float *d_Src = d_Data  + blockIdx_y * N;
          float *d_Dst = d_Data + blockIdx_y * N;
          int lo = pos & (stride - 1);
          int i0 = ((pos - lo) << 2) + lo;
          int i1 = i0 + stride;
          int i2 = i1 + stride;
          int i3 = i2 + stride;

          float D0 = d_Src[i0];
          float D1 = d_Src[i1];
          float D2 = d_Src[i2];
          float D3 = d_Src[i3];

          float T;
          T = D0;
          D0        = D0 + D2;
          D2        = T - D2;
          T = D1;
          D1        = D1 + D3;
          D3        = T - D3;
          T = D0;
          d_Dst[i0] = D0 + D1;
          d_Dst[i1] = T - D1;
          T = D2;
          d_Dst[i2] = D2 + D3;
          d_Dst[i3] = T - D3;
        }
      }
    }
#endif

    {
      float s_data[2048];
      {
        int lid = omp_get_thread_num();
        int gid = omp_get_team_num();
        int gsz = omp_get_num_threads(); 

        // Handle to thread block group
        const int    N = 1 << log2N;
        const int base = gid << log2N;

        const float *d_Src = d_Data + base;
        float *d_Dst = d_Data + base;

        for (int pos = lid; pos < N; pos += gsz)
        {
            s_data[pos] = d_Src[pos];
        }

        //Main radix-4 stages
        const int pos = lid;

        for (int stride = N >> 2; stride > 0; stride >>= 2)
        {
            int lo = pos & (stride - 1);
            int i0 = ((pos - lo) << 2) + lo;
            int i1 = i0 + stride;
            int i2 = i1 + stride;
            int i3 = i2 + stride;

            float D0 = s_data[i0];
            float D1 = s_data[i1];
            float D2 = s_data[i2];
            float D3 = s_data[i3];

            float T;
            T = D0;
            D0         = D0 + D2;
            D2         = T - D2;
            T = D1;
            D1         = D1 + D3;
            D3         = T - D3;
            T = D0;
            s_data[i0] = D0 + D1;
            s_data[i1] = T - D1;
            T = D2;
            s_data[i2] = D2 + D3;
            s_data[i3] = T - D3;
        }

        //Do single radix-2 stage for odd power of two
        if (log2N & 1)
        {

            for (int pos = lid; pos < N / 2; pos += gsz)
            {
                int i0 = pos << 1;
                int i1 = i0 + 1;

                float D0 = s_data[i0];
                float D1 = s_data[i1];
                s_data[i0] = D0 + D1;
                s_data[i1] = D0 - D1;
            }
        }


        for (int pos = lid; pos < N; pos += gsz)
        {
            d_Dst[pos] = s_data[pos];
        }
      }
    }
}

// Modulate two arrays
void modulateGPU(float *__restrict d_A, const float *__restrict d_B, int N)
{
    float     rcpN = 1.0f / (float)N;
    for (int pos = 0; pos < N; pos++)
    {
        d_A[pos] *= d_B[pos] * rcpN;
    }
}
