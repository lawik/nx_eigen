// nx_eigen_fft_eigen.cpp -- FFT via Eigen's bundled (kissfft-backed)
// unsupported/Eigen/FFT module.
//
// Implements the interface declared in nx_eigen_fft.h.
// Linked into the NIF when NX_EIGEN_FFT_LIB=eigen.
//
// Header-only and self-contained: no external FFT library on the
// target. This is the implementation of choice for cross-compiled
// (Nerves) builds where FFTW is unavailable — it replaces the `none`
// stub that made Nx.fft raise on-device.
//
// Eigen::FFT's default (kissfft) scales neither direction when asked
// not to; the nx_eigen contract wants UNNORMALISED forward and inverse
// (the caller divides the inverse by n), so the inverse uses the
// Unscaled flag.

#include "nx_eigen_fft.h"

#include <complex>
#include <unsupported/Eigen/FFT>

namespace {

template <typename T>
int do_fft(const T *in, T *out, int n, bool forward) {
  if (n <= 0)
    return -1;

  try {
    // thread-local instance: Eigen::FFT caches kissfft plans per size
    // internally, so reusing the instance avoids a plan rebuild every
    // transform (NIF calls stay on scheduler threads)
    static thread_local Eigen::FFT<T> fft;
    fft.SetFlag(Eigen::FFT<T>::Unscaled);

    const std::complex<T> *cin = reinterpret_cast<const std::complex<T> *>(in);
    std::complex<T> *cout = reinterpret_cast<std::complex<T> *>(out);

    // Eigen::FFT can't transform in place; the contract allows aliasing.
    std::vector<std::complex<T>> tmp(cin, cin + n);

    if (forward)
      fft.fwd(cout, tmp.data(), n);
    else
      fft.inv(cout, tmp.data(), n);

    return 0;
  } catch (...) {
    return -1;
  }
}

} // namespace

extern "C" {

int nx_eigen_fft_forward_f32(const float *in, float *out, int n) {
  return do_fft<float>(in, out, n, true);
}

int nx_eigen_fft_inverse_f32(const float *in, float *out, int n) {
  return do_fft<float>(in, out, n, false);
}

int nx_eigen_fft_forward_f64(const double *in, double *out, int n) {
  return do_fft<double>(in, out, n, true);
}

int nx_eigen_fft_inverse_f64(const double *in, double *out, int n) {
  return do_fft<double>(in, out, n, false);
}

} // extern "C"
