#include <iostream>
#include <vector>
#include <xsimd/xsimd.hpp>

int main() {
    std::cout << "CoreVector DB - High Performance Vector Database initialized!" << std::endl;
    
    // Simple xsimd test to ensure it's linked correctly
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> res(4);
    
    // For xsimd 11+
    auto va = xsimd::load_unaligned(a.data());
    auto vb = xsimd::load_unaligned(b.data());
    auto vres = va + vb;
    vres.store_unaligned(res.data());
    
    std::cout << "Vector addition test via xsimd: " 
              << res[0] << ", " << res[1] << ", " 
              << res[2] << ", " << res[3] << std::endl;
              
    return 0;
}
