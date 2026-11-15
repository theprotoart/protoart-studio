const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
const int m0 = threadgroup_position_in_grid.y * BM, n0 = threadgroup_position_in_grid.x * BN;
const int tid = thread_position_in_threadgroup.x, warp = tid / 32;
const int wm = (warp / 2) * (BM / 2), wn = (warp % 2) * (BN / 2);
// Four SIMD groups reuse the same input tiles. Half represents every int8 exactly.
threadgroup half A[BM * (BK + 8)], B[BN * (BK + 8)];
simdgroup_float8x8 sums[BM / 16][BN / 16];
int2 total[BM / 16][BN / 16] = {};
for (int k0 = 0; k0 < K; k0 += BK) {
  for (int i = tid; i < BM * (BK / 8); i += 128) {
    int m = i / (BK / 8), k = (i % (BK / 8)) * 8;
    half4 u = 0, v = 0;
    if (m0 + m < M) {
      device const char4 *p = (device const char4 *)(x + (m0 + m) * K + k0 + k);
      u = half4(p[0]);
      v = half4(p[1]);
    }
    *((threadgroup half4 *)(A + m * (BK + 8) + k)) = u;
    *((threadgroup half4 *)(A + m * (BK + 8) + k + 4)) = v;
  }
  for (int i = tid; i < BN * (BK / 8); i += 128) {
    int n = i / (BK / 8), k = (i % (BK / 8)) * 8;
    device const char4 *p = (device const char4 *)(w + (n0 + n) * K + k0 + k);
    *((threadgroup half4 *)(B + n * (BK + 8) + k)) = half4(p[0]);
    *((threadgroup half4 *)(B + n * (BK + 8) + k + 4)) = half4(p[1]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Flush to integer totals every 1024 products, while float32 is still exact.
  if ((k0 % 1024) == 0) {
    for (int a = 0; a < BM / 16; ++a) {
      for (int b = 0; b < BN / 16; ++b) {
        sums[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
      }
    }
  }
  for (int k = 0; k < BK; k += 8) {
    simdgroup_half8x8 af[BM / 16], bf[BN / 16];
    for (int a = 0; a < BM / 16; ++a) {
      simdgroup_load(af[a], A + (wm + 8 * a) * (BK + 8) + k, BK + 8);
    }
    for (int b = 0; b < BN / 16; ++b) {
      simdgroup_load(bf[b], B + (wn + 8 * b) * (BK + 8) + k, BK + 8, ulong2(0), true);
    }
    for (int a = 0; a < BM / 16; ++a) {
      for (int b = 0; b < BN / 16; ++b) {
        simdgroup_multiply_accumulate(sums[a][b], af[a], bf[b], sums[a][b]);
      }
    }
  }
  if (((k0 + BK) % 1024) == 0 || k0 + BK == K) {
    for (int a = 0; a < BM / 16; ++a) {
      for (int b = 0; b < BN / 16; ++b) {
        for (int e = 0; e < 2; ++e) {
          total[a][b][e] += int(sums[a][b].thread_elements()[e]);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Apple SIMD matrix fragments distribute two adjacent columns to each lane.
const int lane = tid % 32;
const int m = m0 + wm + ((lane >> 2) & 4) + ((lane >> 1) & 3);
const int n = n0 + wn + ((lane >> 1) & 4) + (lane & 1) * 2;
for (int a = 0; a < BM / 16; ++a) {
  if (m + a * 8 < M) {
    for (int b = 0; b < BN / 16; ++b) {
      for (int e = 0; e < 2; ++e) {
        y[(m + a * 8) * N + n + b * 8 + e] =
            T(float(total[a][b][e]) * s[m + a * 8] * z[n + b * 8 + e]);
      }
    }
  }
}
