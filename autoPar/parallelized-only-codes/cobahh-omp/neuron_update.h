#include <chrono>
#include <omp.h>

inline int _timestep(float t, float dt)
{
  return (int)((t + 1e-3f*dt)/dt); 
}

void neurongroup_stateupdater (
    float* __restrict  _ptr_array_neurongroup_ge,
    float* __restrict  _ptr_array_neurongroup_gi,
    float* __restrict  _ptr_array_neurongroup_h,
    float* __restrict  _ptr_array_neurongroup_m,
    float* __restrict  _ptr_array_neurongroup_n,
    float* __restrict  _ptr_array_neurongroup_v,
    float* __restrict   _ptr_array_neurongroup_lastspike,
    float* __restrict  _ptr_array_defaultclock_dt,
    float*__restrict  _ptr_array_defaultclock_t,
    char* __restrict  _ptr_array_neurongroup_not_refractory,
    const int _N,
    const int iteration ) 
{
  const float dt = _ptr_array_defaultclock_dt[0];
  const float t = _ptr_array_defaultclock_t[0];
  const int    _lio_1 = _timestep(0.003, dt);
  const float  _lio_2 = 9.939082f; //1.0f*(0.3291372 * exp(1.0f*(0.055555556 * (-0.063f))/0.001f))/0.001f;
  const float  _lio_3 = -55.555556f; //1.0f*(-0.055555556)/0.001f;
  const float  _lio_4 = 0.00001f; //2980.958 * (0.001f * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float  _lio_5 = -200.0f; //1.0f*(-0.2)/0.001f;
  const float  _lio_6 = -0.02016f; //0.32 * (-0.063f);
  const float  _lio_7 = -0.000001f; //- 0.001f * 0.001f;
  const float  _lio_8 = 0; //25.79034 * ((0.001f * 0.001f) * exp(1.0f*(0.25 * (-0.063f))/0.001f));
  const float  _lio_9 = -250.0f; //1.0f*(-0.25)/0.001f;
  const float _lio_10 = 0.00416f; //4.16 * 0.001f;
  const float _lio_11 = 0.02016f; //(-0.32) * (-0.063f);
  const float _lio_12 = -0.01764f; //0.28 * (-0.063f);
  const float _lio_13 = -0.000001f; //(-1.0) * (0.001f * 0.001f);
  const float _lio_14 = 0.000099f; //0.00033546262 * ((0.001f * 0.001f) * exp(1.0f*((-0.2) * (-0.063f))/0.001f));
  const float _lio_15 = 200.0f; //1.0f*0.2/0.001f;
  const float _lio_16 = 0.0112f; //11.2 * 0.001f;
  const float _lio_17 = -0.002016f; //0.032 * (-0.063f);
  const float _lio_18 = 0; //20.085537 * ((0.001f * 0.001f) * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float _lio_19 = 0.00048f; //0.48 * 0.001f;
  const float _lio_20 = 0.002016f; //(-0.032) * (-0.063f);
  const float _lio_21 = 132.901474f; //1.0f*(0.6420127 * exp(1.0f*(0.025 * (-0.063f))/0.001f))/0.001f;
  const float _lio_22 = -25.0f ;//1.0f*(-0.025)/0.001f;
  const float _lio_23 = expf(-2000.0f*dt);
  const float _lio_24 = -3.0f; //1.0f*((-0.06f) * 1e-08f)/2e-10f;
  const float _lio_25 = -2700.0f; //1.0f*((-0.09f) * 6e-06f)/2e-10f;
  const float _lio_26 = 5000.0f; //1.0f*(0.05f * 2e-05f)/2e-10f;
  const float _lio_27 = 0; // 1.0f*0.0f/2e-10f;
  const float _lio_28 = -400000000.0f; //1.0f*(-0.08f)/2e-10f;
  const float _lio_29 = -50.0f; //0.0 - (1.0f*1e-08f/2e-10f);
  const float _lio_30 = -30000.0f; //1.0f*(- 6e-06f)/2e-10f;
  const float _lio_31 = 100000.0f; //1.0f*2e-05f/2e-10f;
  const float _lio_32 = 5000000000.0f; //1.0f*1.0/2e-10f;
  const float _lio_33 = expf(-100.0f*dt);

  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iteration; i++) {
      for(int _idx=0; _idx<_N; _idx++)
      {
        float h = _ptr_array_neurongroup_h[_idx];
        float m = _ptr_array_neurongroup_m[_idx];
        float n = _ptr_array_neurongroup_n[_idx];
        float ge = _ptr_array_neurongroup_ge[_idx];
        float v = _ptr_array_neurongroup_v[_idx];
        const float lastspike = _ptr_array_neurongroup_lastspike[_idx];
        float gi = _ptr_array_neurongroup_gi[_idx];
        char not_refractory;
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        const float _BA_h = (_lio_2 * expf(_lio_3 * v))/(((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * v)))) - (_lio_2 * expf(_lio_3 * v)));
        const float _h = (- _BA_h) + ((_BA_h + h) * expf(dt * (((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * v)))) - (_lio_2 * expf(_lio_3 * v)))));
        const float _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))))/(((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))));
        const float _m = (- _BA_m) + ((_BA_m + m) * expf(dt * (((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))))));
        const float _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))))/(((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_21 * expf(_lio_22 * v))));
        const float _n = (- _BA_n) + ((_BA_n + n) * expf(dt * (((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_21 * expf(_lio_22 * v))))));
        const float _ge = _lio_23 * ge;
        const float _BA_v = (_lio_24 + ((((_lio_25 * (n*n*n*n)) + (_lio_26 * (h * (m*m*m)))) + (_lio_27 * ge)) + (_lio_28 * gi)))/((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)));
        const float _v = (- _BA_v) + ((_BA_v + v) * expf(dt * ((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)))));
        const float _gi = _lio_33 * gi;

        _ptr_array_neurongroup_h[_idx] = _h;
        _ptr_array_neurongroup_m[_idx] = _m;
        _ptr_array_neurongroup_n[_idx] = _n;
        _ptr_array_neurongroup_ge[_idx] = _ge;
        _ptr_array_neurongroup_v[_idx] = _v;
        _ptr_array_neurongroup_gi[_idx] = _gi;
        _ptr_array_neurongroup_not_refractory[_idx] = not_refractory;
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / iteration);
  }
}
