defmodule NxEigen do
  @moduledoc """
  An Elixir Nx backend that binds the Eigen C++ library for efficient numerical computing.

  NxEigen provides high-performance linear algebra operations optimized for embedded systems,
  particularly targeting the Arduino Uno Q. It implements the complete Nx.Backend behavior,
  allowing seamless integration with the Nx ecosystem.

  ## Features

  - **Complete Nx.Backend implementation** - All required callbacks implemented
  - **Efficient linear algebra** - Uses Eigen's optimized matrix operations
  - **FFT support** - Fast Fourier transforms via pluggable interface (FFTW3 by default)
  - **All Nx types** - Support for u8-u64, s8-s64, f32/f64, c64/c128
  - **Embedded-friendly** - Bitwise operations, integer math, and efficient memory usage

  ## Usage

      # Create tensors with the NxEigen backend
      t = NxEigen.tensor([[1, 2], [3, 4]])

      # All Nx operations work automatically
      result = Nx.dot(t, t)

      # Matrix operations use Eigen's optimized routines
      a = NxEigen.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
      b = Nx.transpose(a)
      result = Nx.dot(a, b)

      # FFT operations
      fft_result = Nx.fft(NxEigen.tensor([1.0, 0.0, 0.0, 0.0]), length: 4)

  See the README for detailed installation instructions, platform support, and
  cross-compilation guides.
  """

  @doc """
  Returns a new tensor with the given data using the NxEigen backend.

  ## Options

  All standard Nx.tensor/2 options are supported except `:backend`, which is
  automatically set to `NxEigen.Backend`.

  ## Examples

      iex> NxEigen.tensor([1, 2, 3])
      #Nx.Tensor<
        s32[3]
        [1, 2, 3]
      >

      iex> NxEigen.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 2.0],
          [3.0, 4.0]
        ]
      >
  """
  def tensor(data, opts \\ []) do
    Nx.tensor(data, Keyword.put(opts, :backend, NxEigen.Backend))
  end
end
