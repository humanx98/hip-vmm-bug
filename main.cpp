// Reproduces a ROCm virtual memory management bug: hipMemUnmap() + hipMemMap() of another allocation at the same
// virtual address does not invalidate the GPU TLB, so the address keeps reading the old physical memory.
// Seen on Ubuntu 26.04 with Linux 7.0.0, TheRock ROCm 10.1.0, AMD Radeon RX 6800M (gfx1031).
//
// Workaround: map any host memory for the GPU (hipHostMalloc + hipHostFree). That goes through the kernel driver,
// which invalidates the TLB.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK( call )                                                                                   \
    do                                                                                                  \
    {                                                                                                   \
        const hipError_t error = ( call );                                                              \
        if( error != hipSuccess )                                                                       \
        {                                                                                               \
            std::printf( "%s failed: %s\n", #call, hipGetErrorString( error ) );                        \
            std::exit( 1 );                                                                             \
        }                                                                                               \
    } while( 0 )

constexpr size_t pageBytes = 64 * 1024;

// The byte the whole page holds, read back to the host.
unsigned char readPage( const unsigned char* page )
{
    std::vector<unsigned char> host( pageBytes );
    CHECK( hipMemcpy( host.data(), page, pageBytes, hipMemcpyDeviceToHost ) );
    return host[0];
}

void report( const char* step, const unsigned char* page0, const unsigned char* page1 )
{
    const unsigned char value0 = readPage( page0 );
    const unsigned char value1 = readPage( page1 );
    const bool correct = value0 == 0xBB && value1 == 0xAA;
    std::printf( "%-28s page0 = 0x%02X, page1 = 0x%02X  (expected 0xBB, 0xAA) -> %s\n", step, value0, value1,
                 correct ? "OK" : "STALE" );
}

int main()
{
    int device = 0;
    CHECK( hipGetDevice( &device ) );
    hipDeviceProp_t properties{};
    CHECK( hipGetDeviceProperties( &properties, device ) );
    std::printf( "GPU: %s (%s)\n", properties.name, properties.gcnArchName );

    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = device;
    access.flags         = hipMemAccessFlagsProtReadWrite;

    // Two virtual pages and two physical allocations A and B.
    unsigned char* page0 = nullptr;
    CHECK( hipMemAddressReserve( reinterpret_cast<void**>( &page0 ), 2 * pageBytes, pageBytes, nullptr, 0 ) );
    unsigned char* page1 = page0 + pageBytes;

    hipMemGenericAllocationHandle_t a = nullptr, b = nullptr;
    CHECK( hipMemCreate( &a, pageBytes, &prop, 0 ) );
    CHECK( hipMemCreate( &b, pageBytes, &prop, 0 ) );

    // page0 -> A = 0xAA, page1 -> B = 0xBB.
    CHECK( hipMemMap( page0, pageBytes, 0, a, 0 ) );
    CHECK( hipMemMap( page1, pageBytes, 0, b, 0 ) );
    CHECK( hipMemSetAccess( page0, 2 * pageBytes, &access, 1 ) );
    CHECK( hipMemset( page0, 0xAA, pageBytes ) );
    CHECK( hipMemset( page1, 0xBB, pageBytes ) );
    CHECK( hipDeviceSynchronize() );
    std::printf( "%-28s page0 = 0x%02X, page1 = 0x%02X\n", "initial mapping", readPage( page0 ), readPage( page1 ) );

    // Swap: page0 -> B, page1 -> A. From here on page0 must read 0xBB and page1 0xAA.
    CHECK( hipMemUnmap( page0, pageBytes ) );
    CHECK( hipMemUnmap( page1, pageBytes ) );
    CHECK( hipMemMap( page0, pageBytes, 0, b, 0 ) );
    CHECK( hipMemMap( page1, pageBytes, 0, a, 0 ) );
    CHECK( hipMemSetAccess( page0, 2 * pageBytes, &access, 1 ) );
    CHECK( hipDeviceSynchronize() );
    report( "after swapping mappings", page0, page1 );

    // Workaround: mapping host memory for the GPU makes the driver invalidate the TLB.
    void* hostMemory = nullptr;
    CHECK( hipHostMalloc( &hostMemory, 4096 ) );
    CHECK( hipHostFree( hostMemory ) );
    report( "after hipHostMalloc/Free", page0, page1 );

    CHECK( hipMemUnmap( page0, pageBytes ) );
    CHECK( hipMemUnmap( page1, pageBytes ) );
    CHECK( hipMemRelease( a ) );
    CHECK( hipMemRelease( b ) );
    CHECK( hipMemAddressFree( page0, 2 * pageBytes ) );
    return 0;
}
