#include "morphology.h"

enum class MorphOpType {
    ERODE,
    DILATE,
};


template <MorphOpType opType>
inline unsigned char elementOp(unsigned char lhs, unsigned char rhs)
{
}

template <>
inline unsigned char elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <>
inline unsigned char elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
    return lhs > rhs ? lhs : rhs;
}

template <MorphOpType opType>
inline unsigned char borderValue()
{
}

template <>
inline unsigned char borderValue<MorphOpType::ERODE>()
{
    return BLACK;
}

template <>
inline unsigned char borderValue<MorphOpType::DILATE>()
{
    return WHITE;
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType>
void twoWayScan(unsigned char* __restrict buffer,
                unsigned char* __restrict opArray,
                const int selSize,
                const int tid)
{
  opArray[tid] = buffer[tid];
  opArray[tid + selSize] = buffer[tid + selSize];

  for (int offset = 1; offset < selSize; offset *= 2) {
    if (tid >= offset) {
      opArray[tid + selSize - 1] = 
        elementOp<opType>(opArray[tid + selSize - 1], opArray[tid + selSize - 1 - offset]);
    }
    if (tid <= selSize - 1 - offset) {
      opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
    }
  }
}


template <MorphOpType opType>
double morphology(unsigned char* img_d,
                unsigned char* tmp_d,
                const int width,
                const int height,
                const int hsize,
                const int vsize)
{
  int blockSize_x_h = hsize;
  int blockSize_y_h = 1;
  int gridSize_x_h = roundUp(width, blockSize_x_h);
  int gridSize_y_h = roundUp(height, blockSize_y_h);

  int blockSize_x_v = 1;
  int blockSize_y_v = vsize;
  int gridSize_x_v = roundUp(width, blockSize_x_v);
  int gridSize_y_v = roundUp(height, blockSize_y_v);

  // initialize the intermediate buffer may be required
  unsigned int memSize = width * height * sizeof(unsigned char);
  memset(tmp_d, 0, memSize);
  
  auto start = std::chrono::steady_clock::now();
  
  {
    //size_t sMemSize = 4 * hsize * sizeof(unsigned char);
    unsigned char sMem[128];
    {
      unsigned char* buffer = sMem;
      unsigned char* opArray = buffer + 2 * hsize;

      int bx = omp_get_team_num() % gridSize_x_h;
      int by = omp_get_team_num() / gridSize_x_h;
      int tx = omp_get_thread_num();   // blockSize_y_h = 1 so ty = 0

      const int tidx = tx + bx * blockSize_x_h;
      const int tidy =      by * blockSize_y_h;
      if (tidx < width && tidy < height) {

        buffer[tx] = img_d[tidy * width + tidx];
        if (tidx + hsize < width) {
            buffer[tx + hsize] = img_d[tidy * width + tidx + hsize];
        }

        twoWayScan<opType>(buffer, opArray, hsize, tx);

        if (tidx + hsize/2 < width - hsize/2) {
            tmp_d[tidy * width + tidx + hsize/2] = 
               elementOp<opType>(opArray[tx], opArray[tx + hsize - 1]);
        }
      }
    }
  }

  {
    //size_t sMemSize = 4 * hsize * sizeof(unsigned char);
    unsigned char sMem[128];
    {
      unsigned char* buffer = sMem;
      unsigned char* opArray = buffer + 2 * vsize;

      int bx = omp_get_team_num() % gridSize_x_v;
      int by = omp_get_team_num() / gridSize_x_v;
      int ty = omp_get_thread_num();   // blockSize_x_v = 1 so tx = 0

      const int tidx =      bx * blockSize_x_v;
      const int tidy = ty + by * blockSize_y_v;
      if (tidx < width && tidy < height) {

        buffer[ty] = tmp_d[tidy * width + tidx];
        if (tidy + vsize < height) {
            buffer[ty + vsize] = tmp_d[(tidy + vsize) * width + tidx];
        }

        twoWayScan<opType>(buffer, opArray, vsize, ty);

        if (tidy + vsize/2 < height - vsize/2) {
            img_d[(tidy + vsize/2) * width + tidx] = 
                elementOp<opType>(opArray[ty], opArray[ty + vsize - 1]);
        }

        if (tidy < vsize/2 || tidy >= height - vsize/2) {
            img_d[tidy * width + tidx] = borderValue<opType>();
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return time;
}

extern "C"
double erode(unsigned char* img_d,
             unsigned char* tmp_d,
             const int width,
             const int height,
             const int hsize,
             const int vsize)
{
  return morphology<MorphOpType::ERODE>(img_d, tmp_d, width, height, hsize, vsize);
}

extern "C"
double dilate(unsigned char* img_d,
              unsigned char* tmp_d,
              const int width,
              const int height,
              const int hsize,
              const int vsize)
{
  return morphology<MorphOpType::DILATE>(img_d, tmp_d, width, height, hsize, vsize);
}
