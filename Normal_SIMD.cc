#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include<bits/stdc++.h>
#include <algorithm>
#include <omp.h>
#include <arm_neon.h>
typedef long long int ll;
const int g = 3;
// 可以自行添加需要的头文件
alignas(32) ll rev[300000]={0};
static inline uint64x2_t vmulq_u64(uint64x2_t a, uint64x2_t b) {
    uint64x2_t result;
    result = vsetq_lane_u64(vgetq_lane_u64(a, 0) * vgetq_lane_u64(b, 0), result, 0);
    result = vsetq_lane_u64(vgetq_lane_u64(a, 1) * vgetq_lane_u64(b, 1), result, 1);
    return result;
}
void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    //std::cout<<"astart";
    //fout<<"astart";
    for (int i = 0; i < *n; i++){
        fin>>a[i];
      //  std::cout<<a[i];
    }
    //std::cout<<"bstart";
    //fout<<"bstart";
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
        //std::cout<<b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    //std::cout<<"abstart!";
    //fout<<"abstart";
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
      //  std::cout<<ab[i]<<"\n";
    }
}


struct Montgomery32{    
    uint32_t m,ir;
    static uint32_t inv_m(uint32_t m){
        uint32_t x =1;
        for(uint32_t i = 0;i<5;i++)
            x*=2-x*m;
    return x;}
    Montgomery32(uint32_t m_) : m(m_) {
        uint32_t inv = inv_m(m);
        ir = uint32_t(-int32_t(inv));
      }
};

struct MontgomeryNEON{
        uint32x4_t vm;   // [m, m, m, m]
        uint32x4_t vir;  // [ir,ir,ir,ir]
        uint64x2_t vm64; // 同步长的 u64 版本 m
        uint32x4_t vR2; 


    MontgomeryNEON(uint32_t m) {
        Montgomery32 M(m);
        vm   = vdupq_n_u32(M.m);
        vir  = vdupq_n_u32(M.ir);
        vm64 = vdupq_n_u64(uint64_t(M.m));
        uint32_t Rmod  = (uint64_t(1) << 32) % m;
        uint32_t R2mod = (uint64_t(Rmod) * Rmod) % m;
        vR2  = vdupq_n_u32(R2mod);
      }

    inline uint32x4_t mul4(uint32x4_t a4, uint32x4_t b4) const {
        // 1. 使用混合宽度乘法
        uint32x2_t a_lo = vget_low_u32(a4);
        uint32x2_t a_hi = vget_high_u32(a4);
        uint32x2_t b_lo = vget_low_u32(b4);
        uint32x2_t b_hi = vget_high_u32(b4);
        
        // 执行32x32->64位乘法
        uint64x2_t t_lo = vmull_u32(a_lo, b_lo);
        uint64x2_t t_hi = vmull_u32(a_hi, b_hi);
        
        // 2. 直接从64位结果提取低32位，减少vmovn操作
        uint32x2_t t_lo32 = vmovn_u64(t_lo);
        uint32x2_t t_hi32 = vmovn_u64(t_hi);
        
        // 3. 使用vqdmulh/vqrdmulh等指令可以在某些情况下取代普通乘法
        // 获取ir的低位部分
        uint32x2_t ir_lo = vget_low_u32(vir);
        
        // 计算u值 - 使用向量乘法
        uint32x2_t u_lo = vmul_u32(t_lo32, ir_lo);
        uint32x2_t u_hi = vmul_u32(t_hi32, ir_lo);
        
        // 4. 合并中间计算，减少寄存器压力
        // 将u扩展到64位并乘以m
        uint64x2_t um_lo = vmull_u32(u_lo, vget_low_u32(vm));
        uint64x2_t um_hi = vmull_u32(u_hi, vget_high_u32(vm));
        
        // 5. 使用vmlal_u32代替单独的乘法和加法
        // 将um加到t上，可以在一条指令中完成
        uint64x2_t r_lo = vaddq_u64(t_lo, um_lo);
        uint64x2_t r_hi = vaddq_u64(t_hi, um_hi);
        
        // 6. 优化右移操作
        // 使用逻辑右移
        r_lo = vshrq_n_u64(r_lo, 32);
        r_hi = vshrq_n_u64(r_hi, 32);
        
        // 7. 使用vzipq重排数据，可能比单独的vmovn和vcombine更高效
        uint32x2_t r_lo32 = vmovn_u64(r_lo);
        uint32x2_t r_hi32 = vmovn_u64(r_hi);
        
        return vcombine_u32(r_lo32, r_hi32);
    }
    // 最后如果需要把结果规约到 [0,m)：
    inline uint32x4_t normalize4(uint32x4_t x4) const {
        // 如果 x >= m，则减 m
        uint32x4_t mask = vcgeq_u32(x4, vm);     // x4>=m ? 0xFFFFFFFF : 0
        uint32x4_t sub  = vandq_u32(mask, vm);   // 若需减就取 m，否则 0
        return vsubq_u32(x4, sub);
    }

    // 向量版 Montgomery 域转换： tran(a) = a * R mod m
    inline uint32x4_t tran4(uint32x4_t x4) const {
        // REDC( x * R^2 ) == a * R mod m
        uint32x4_t t = mul4(x4, vR2);
        return normalize4(t);
    }

    // 向量版 Montgomery 还原： val(a) = a * R^{-1} mod m
    inline uint32x4_t val4(uint32x4_t x4) const {
        // REDC( x * 1 ) == a * R^{-1} mod m
        uint32x4_t t = mul4(x4, vdupq_n_u32(1));
        return normalize4(t);
    }
    inline void butterfly4(uint32x4_t &a_vec, uint32x4_t &b_vec, uint32x4_t w_vec) const {
        // 乘法: wb = w * b
        uint32x4_t wb_vec = mul4(w_vec, b_vec);
        wb_vec = normalize4(wb_vec);
        
        // 保存原始a_vec用于计算差
        uint32x4_t a_orig = a_vec;
        
        // 直接更新a_vec = a + wb
        a_vec = vaddq_u32(a_vec, wb_vec);
        a_vec = normalize4(a_vec);
        
        // 直接更新b_vec = a - wb
        b_vec = vsubq_u32(vaddq_u32(a_orig, vm), wb_vec);
        b_vec = normalize4(b_vec);
    }
};


int pow(int x,int y,int p)//这个函数用于NTT的快速幂计算
{  
	ll z=1ll*x;
    ll ans=1ll;
	for (;y;y/=2,z=z*z%p)
        if (y&1)
            ans=ans*z%p;
	return (int)ans%p;
}

inline int fast_pow(int x, int y, int p) {
    int result = 1;
    int base = x;
    
    while (y > 0) {
        if (y & 1) result = (1LL * result * base) % p;
        base = (1LL * base * base) % p;
        y >>= 1;
    }
    
    return result;
}

alignas(32) static uint32_t wn_powers_forward[20][262144]; // 正变换旋转因子
alignas(32) static uint32_t wn_powers_inverse[20][262144]; // 逆变换旋转因子
alignas(32) static bool wn_precomputed_forward[20] = {false};
alignas(32) static bool wn_precomputed_inverse[20] = {false};

// 优化的NTT实现
void ntt_simd_optimized(int *a, int n, int p, int inv) {
    int lg = 0;
    while ((1 << lg) < n) lg++;
    
    // 位反转置换保持不变
    memset(rev, 0, n * sizeof(rev[0]));
    for (int i = 0; i < n; i++) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }
    
    const int SIMD_WIDTH = 4;
    MontgomeryNEON mont(p);
    
    // 批量转换到Montgomery域
    for (int i = 0; i < n; i += SIMD_WIDTH) {
        if (i + SIMD_WIDTH <= n) {
            uint32x4_t a_vec = vld1q_u32((uint32_t*)(a + i));
            a_vec = mont.tran4(a_vec);
            vst1q_u32((uint32_t*)(a + i), a_vec);
        } else {
            for (int j = i; j < n; j++) {
                uint32x4_t a_vec = vdupq_n_u32(a[j]);
                a_vec = mont.tran4(a_vec);
                a[j] = vgetq_lane_u32(a_vec, 0);
            }
        }
    }
    
    // 预计算所有层级的旋转因子 - 使用分离的缓存
    for (int s = 1; s <= lg; s++) {
        int mid = 1 << (s - 1);
        
        // 选择正确的旋转因子缓存
        uint32_t* wn_powers;
        bool* wn_precomputed_flag;
        
        if (inv == 1) {
            wn_powers = wn_powers_forward[s];
            wn_precomputed_flag = &wn_precomputed_forward[s];
        } else {
            wn_powers = wn_powers_inverse[s];
            wn_precomputed_flag = &wn_precomputed_inverse[s];
        }
        
        // 如果该层级的旋转因子尚未预计算，就计算并缓存
        if (!(*wn_precomputed_flag)) {
            int wn = fast_pow(g, (p - 1) / (mid * 2), p);
            if (inv == -1) wn = fast_pow(wn, p - 2, p);
            
            // 计算旋转因子表
            wn_powers[0] = 1;
            
            // 分块计算旋转因子，提高缓存局部性
            const int BLOCK_SIZE = 16; // 缓存友好的块大小
            for (int block = 0; block < mid; block += BLOCK_SIZE) {
                int end = block + BLOCK_SIZE < mid ? block + BLOCK_SIZE : mid;
                
                for (int i = block; i < end; i++) {
                    if (i == 0) continue;
                    wn_powers[i] = (1LL * wn_powers[i-1] * wn) % p;
                }
            }
            
            // 使用SIMD批量转换旋转因子到Montgomery域
            for (int i = 0; i < mid; i += SIMD_WIDTH) {
                if (i + SIMD_WIDTH <= mid) {
                    uint32x4_t w_vec = vld1q_u32((uint32_t*)(wn_powers + i));
                    w_vec = mont.tran4(w_vec);
                    vst1q_u32((uint32_t*)(wn_powers + i), w_vec);
                } else {
                    for (int j = i; j < mid; j++) {
                        uint32x4_t w_vec = vdupq_n_u32(wn_powers[j]);
                        w_vec = mont.tran4(w_vec);
                        wn_powers[j] = vgetq_lane_u32(w_vec, 0);
                    }
                }
            }
            
            *wn_precomputed_flag = true;
        }
        
        // 按照要求进行蝴蝶操作优化
        // 当步长小于SIMD宽度时，使用朴素方法
        if (mid < SIMD_WIDTH) {
            for (int i = 0; i < n; i += mid * 2) {
                for (int j = 0; j < mid; j++) {
                    uint32x4_t a_mont = vdupq_n_u32(a[i + j]);
                    uint32x4_t b_mont = vdupq_n_u32(a[i + j + mid]);
                    uint32x4_t w_mont = vdupq_n_u32(wn_powers[j]);
                    
                    mont.butterfly4(a_mont, b_mont, w_mont);
                    
                    a[i + j] = vgetq_lane_u32(a_mont, 0);
                    a[i + j + mid] = vgetq_lane_u32(b_mont, 0);
                }
            }
        } 
        // 当步长大于等于SIMD宽度时，使用SIMD优化
        else {
            for (int i = 0; i < n; i += mid * 2) {
                for (int j = 0; j < mid; j += SIMD_WIDTH) {
                    if (j + SIMD_WIDTH <= mid) {
                        // SIMD批量处理
                        uint32x4_t a_vec = vld1q_u32((uint32_t*)(a + i + j));
                        uint32x4_t b_vec = vld1q_u32((uint32_t*)(a + i + j + mid));
                        uint32x4_t w_vec = vld1q_u32((uint32_t*)(wn_powers + j));
                        
                        mont.butterfly4(a_vec, b_vec, w_vec);
                        
                        vst1q_u32((uint32_t*)(a + i + j), a_vec);
                        vst1q_u32((uint32_t*)(a + i + j + mid), b_vec);
                    } else {
                        // 处理剩余元素
                        for (int k = j; k < mid; k++) {
                            uint32x4_t a_mont = vdupq_n_u32(a[i + k]);
                            uint32x4_t b_mont = vdupq_n_u32(a[i + k + mid]);
                            uint32x4_t w_mont = vdupq_n_u32(wn_powers[k]);
                            
                            mont.butterfly4(a_mont, b_mont, w_mont);
                            
                            a[i + k] = vgetq_lane_u32(a_mont, 0);
                            a[i + k + mid] = vgetq_lane_u32(b_mont, 0);
                        }
                    }
                }
            }
        }
    }
    
    // 逆变换需要乘以n的逆元
    if (inv == -1) {
        int n_inv = fast_pow(n, p - 2, p);
        uint32x4_t n_inv_mont = mont.tran4(vdupq_n_u32(n_inv));
        
        for (int i = 0; i < n; i += SIMD_WIDTH) {
            if (i + SIMD_WIDTH <= n) {
                uint32x4_t a_vec = vld1q_u32((uint32_t*)(a + i));
                a_vec = mont.mul4(a_vec, n_inv_mont);
                a_vec = mont.normalize4(a_vec);
                vst1q_u32((uint32_t*)(a + i), a_vec);
            } else {
                for (int j = i; j < n; j++) {
                    uint32x4_t a_vec = vdupq_n_u32(a[j]);
                    a_vec = mont.mul4(a_vec, n_inv_mont);
                    a_vec = mont.normalize4(a_vec);
                    a[j] = vgetq_lane_u32(a_vec, 0);
                }
            }
        }
    }
    
    // 一次性从Montgomery域转回普通域
    for (int i = 0; i < n; i += SIMD_WIDTH) {
        if (i + SIMD_WIDTH <= n) {
            uint32x4_t a_vec = vld1q_u32((uint32_t*)(a + i));
            a_vec = mont.val4(a_vec);
            vst1q_u32((uint32_t*)(a + i), a_vec);
        } else {
            for (int j = i; j < n; j++) {
                uint32x4_t a_vec = vdupq_n_u32(a[j]);
                a_vec = mont.val4(a_vec);
                a[j] = vgetq_lane_u32(a_vec, 0);
            }
        }
    }
}

// 优化的多项式乘法函数
void poly_multiply_optimized(int *a, int *b, int *ab, int n, int p) {
    int m = 1;
    while (m < 2*n-1) m <<= 1;  // 找到≥2n-1的最小2次幂
    // 填充0
    memset(a + n, 0, (m - n) * sizeof(int));
    memset(b + n, 0, (m - n) * sizeof(int));
    // 使用优化的NTT
    ntt_simd_optimized(a, m, p, 1);
    ntt_simd_optimized(b, m, p, 1);
    
    // SIMD优化点乘过程
    const int SIMD_WIDTH = 4;
    MontgomeryNEON mont(p);
    
    // 批量点乘 - 使用分块提高缓存命中率
    const int BLOCK_SIZE = 128;
    for (int block = 0; block < m; block += BLOCK_SIZE) {
        int end = block + BLOCK_SIZE < m ? block + BLOCK_SIZE : m;
        
        // 预取下一个数据块
        if (block + 2*BLOCK_SIZE < m) {
            __builtin_prefetch(&a[block + 2*BLOCK_SIZE], 0,3);
            __builtin_prefetch(&b[block + 2*BLOCK_SIZE], 0,3);
        }
        
        for (int i = block; i < end; i += SIMD_WIDTH) {
            if (i + SIMD_WIDTH <= m) {
                // 直接加载数据到SIMD寄存器
                uint32x4_t a_vec = vld1q_u32((uint32_t*)(a + i));
                uint32x4_t b_vec = vld1q_u32((uint32_t*)(b + i));
                
                // 转入Montgomery域、点乘、转出
                a_vec = mont.tran4(a_vec);
                b_vec = mont.tran4(b_vec);
                uint32x4_t ab_vec = mont.mul4(a_vec, b_vec);
                ab_vec = mont.normalize4(ab_vec);
                ab_vec = mont.val4(ab_vec);
                
                // 直接存储结果
                vst1q_u32((uint32_t*)(ab + i), ab_vec);
            } else {
                // 处理尾部边界
                for (int j = i; j < end; j++) {
                    uint32x4_t a_vec = vdupq_n_u32(a[j]);
                    uint32x4_t b_vec = vdupq_n_u32(b[j]);
                    a_vec = mont.tran4(a_vec);
                    b_vec = mont.tran4(b_vec);
                    uint32x4_t ab_vec = mont.mul4(a_vec, b_vec);
                    ab_vec = mont.normalize4(ab_vec);
                    ab_vec = mont.val4(ab_vec);
                    ab[j] = vgetq_lane_u32(ab_vec, 0);
                }
            }
        }
    }
    
    // 逆变换
    ntt_simd_optimized(ab, m, p, -1);
    
    // 确保结果都是非负的
    for (int i = 0; i < 2*n-1; ++i) {
        ab[i] = ((ab[i] % p) + p) % p;
    }
}


alignas(32) int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab,0,sizeof(ab));
        memset(rev,0,sizeof(rev));
        memset(wn_precomputed_forward, false, sizeof(wn_precomputed_forward));
        memset(wn_precomputed_inverse, false, sizeof(wn_precomputed_inverse));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply_optimized(a,b,ab,n_,p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    return 0;
}