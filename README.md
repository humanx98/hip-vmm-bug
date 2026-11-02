# hip_vmm_bug

Minimal reproducer: `hipMemUnmap()` followed by `hipMemMap()` of another allocation at the same virtual address
leaves the GPU TLB stale, so the address keeps reading the old physical memory. Mapping any host memory for the GPU
(`hipHostMalloc` + `hipHostFree`) forces the driver to invalidate the TLB.

Seen on Ubuntu 26.04 with Linux 7.0.0, TheRock ROCm 10.1.0, AMD Radeon RX 6800M (gfx1031).

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build
./build/hip_vmm_bug
```

Expected output on an affected system:

```
initial mapping              page0 = 0xAA, page1 = 0xBB
after swapping mappings      page0 = 0xAA, page1 = 0xBB  (expected 0xBB, 0xAA) -> STALE
after hipHostMalloc/Free     page0 = 0xBB, page1 = 0xAA  (expected 0xBB, 0xAA) -> OK
```
