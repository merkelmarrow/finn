# FINN C++ Utilities for NumPy I/O

This directory contains C++ utilities for reading and writing NumPy files in HLS simulations.

## Files

- `npy2apintstream.hpp` - Convert NumPy arrays to ap_int streams
- `npy2vectorstream.hpp` - Convert NumPy arrays to HLS vector streams
- `cnpy.h`, `cnpy.cpp` - NumPy file I/O library (MIT License)

## cnpy Library

The `cnpy.h` and `cnpy.cpp` files are derived from [rogersce/cnpy](https://github.com/rogersce/cnpy)
(MIT License, Copyright Carl Rogers 2011).

This version has been modified by Advanced Micro Devices, Inc. (Yaman Umuroglu) to add support
for the Vitis HLS `half` datatype (float16). AMD modifications are licensed under BSD-3-Clause.

See CNPY_LICENSE for the full MIT license text.

### Modifications from Original

- Added `#include "ap_int.h"` to support HLS datatypes
- Added `half` type support in `map_type()` function for float16 compatibility
