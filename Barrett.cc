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

alignas(32) static bool wn_precomputed_forward[20] = {false};
alignas(32) static bool wn_precomputed_inverse[20] = {false};

// Barrett 归约结构体
struct BarrettReduction {
    uint32_t m;     // 模数
    uint32_t k;     // log2(m) 向上取整
    uint64_t mu;    // floor(2^{2k} / m)
    
    BarrettReduction(uint32_t m_) : m(m_) {
        // 确保 m != 0
        assert(m != 0);
        // 计算 k = ceiling(log2(m))
        k = 32 - __builtin_clz(m - 1);
        // 计算 mu = floor(2^{2k} / m)
        mu = (__uint128_t(1) << (k + k)) / m;
    }
    
    // 计算 a mod m 使用 Barrett 归约
    inline uint32_t reduce(uint64_t a) const {
        if (a < m) return a; // 快速路径
        // 使用 128 位乘法计算 q = floor(a * mu / 2^{2k})
        __uint128_t product = (__uint128_t)a * mu;
        uint64_t q = product >> (k + k);
        // r = a - q * m
        uint32_t r = a - q * m;
        // 可能需要一次额外的减法
        return r >= m ? r - m : r;
    }
    
    // 计算 (a * b) mod m
    inline uint32_t mul_mod(uint32_t a, uint32_t b) const {
        return reduce((__uint128_t)a * b);
    }
};


// NEON 版本的 Barrett 归约
struct BarrettNEON {
    uint32_t m;
    uint32_t k;
    uint64_t mu;
    uint32x4_t vm;
    
    BarrettNEON(uint32_t m_) : m(m_) {
        BarrettReduction br(m);
        k = br.k;
        mu = br.mu;
        vm = vdupq_n_u32(m);
    }
    
    // 向量化的 Barrett 归约
    inline uint32x4_t reduce4(uint32x4_t a) const {
        // 分量级 Barrett 归约
        BarrettReduction br(m);
        uint32_t a0 = vgetq_lane_u32(a, 0);
        uint32_t a1 = vgetq_lane_u32(a, 1);
        uint32_t a2 = vgetq_lane_u32(a, 2);
        uint32_t a3 = vgetq_lane_u32(a, 3);
        
        uint32_t r0 = br.reduce(a0);
        uint32_t r1 = br.reduce(a1);
        uint32_t r2 = br.reduce(a2);
        uint32_t r3 = br.reduce(a3);
        
        return vcombine_u32(vcreate_u32((uint64_t)r1 << 32 | r0),
                           vcreate_u32((uint64_t)r3 << 32 | r2));
    }
    
    // 向量化的模乘法
    inline uint32x4_t mul_mod4(uint32x4_t a, uint32x4_t b) const {
        // 分离低32位乘法
        uint32x2_t a_lo = vget_low_u32(a);
        uint32x2_t a_hi = vget_high_u32(a);
        uint32x2_t b_lo = vget_low_u32(b);
        uint32x2_t b_hi = vget_high_u32(b);
        
        // 执行32x32->64位乘法
        uint64x2_t prod_lo = vmull_u32(a_lo, b_lo);
        uint64x2_t prod_hi = vmull_u32(a_hi, b_hi);
        
        // 提取结果并应用 Barrett 归约
        uint32_t res0 = BarrettReduction(m).reduce(vgetq_lane_u64(prod_lo, 0));
        uint32_t res1 = BarrettReduction(m).reduce(vgetq_lane_u64(prod_lo, 1));
        uint32_t res2 = BarrettReduction(m).reduce(vgetq_lane_u64(prod_hi, 0));
        uint32_t res3 = BarrettReduction(m).reduce(vgetq_lane_u64(prod_hi, 1));
        
        return vcombine_u32(vcreate_u32((uint64_t)res1 << 32 | res0),
                           vcreate_u32((uint64_t)res3 << 32 | res2));
    }
    
    // 向量化的模加法
    inline uint32x4_t add_mod4(uint32x4_t a, uint32x4_t b) const {
        uint32x4_t sum = vaddq_u32(a, b);
        uint32x4_t mask = vcgeq_u32(sum, vm);
        return vsubq_u32(sum, vandq_u32(mask, vm));
    }
    
    // 向量化的模减法
    inline uint32x4_t sub_mod4(uint32x4_t a, uint32x4_t b) const {
        uint32x4_t diff = vaddq_u32(vsubq_u32(a, b), vdupq_n_u32(m));
        uint32x4_t mask = vcgeq_u32(diff, vm);
        return vsubq_u32(diff, vandq_u32(mask, vm));
    }
};

// 修改 ntt_dif_simd 函数使用 Barrett 归约
void ntt_dif_barrett(int *a, int n, int p) {
    const int W = 4;
    BarrettNEON bar(p);
    BarrettReduction br(p);

    int lg = 0; while ((1<<lg) < n) lg++;
    for (int s = lg; s >= 1; --s) {
        int len = 1<<s, mid = len>>1;
        // wlen = g^{(p-1)/len} mod p
        int wlen = fast_pow(g, (p-1)/len, p);
        // 预算 wlen^j
        std::vector<uint32_t> wp(mid);
        wp[0]=1;
        for (int j=1; j<mid; ++j) wp[j] = br.mul_mod(wp[j-1], wlen);

        for (int i=0; i<n; i+=len) {__builtin_prefetch(a + i + len, 0, 3);
            for (int j = 0; j < mid; j += 4*W) {
                // 处理4组数据 - 重排指令减少依赖等待
                if (j + 4*W <= mid) {
                    __builtin_prefetch(a + i + j + 8*W, 0, 2);
                __builtin_prefetch(a + i + j + mid + 8*W, 0, 2);
                    // 预加载4组数据
                    uint32x4_t va0 = vld1q_u32((uint32_t*)(a+i+j));
                    uint32x4_t vb0 = vld1q_u32((uint32_t*)(a+i+j+mid));
                    uint32x4_t va1 = vld1q_u32((uint32_t*)(a+i+j+W));
                    uint32x4_t vb1 = vld1q_u32((uint32_t*)(a+i+j+W+mid));
                    uint32x4_t va2 = vld1q_u32((uint32_t*)(a+i+j+2*W));
                    uint32x4_t vb2 = vld1q_u32((uint32_t*)(a+i+j+2*W+mid));
                    uint32x4_t va3 = vld1q_u32((uint32_t*)(a+i+j+3*W));
                    uint32x4_t vb3 = vld1q_u32((uint32_t*)(a+i+j+3*W+mid));
                    
                    // 预加载旋转因子
                    uint32x4_t w0 = vld1q_u32(wp.data()+j);
                    uint32x4_t w1 = vld1q_u32(wp.data()+j+W);
                    uint32x4_t w2 = vld1q_u32(wp.data()+j+2*W);
                    uint32x4_t w3 = vld1q_u32(wp.data()+j+3*W);
                    
                    // 计算和差
                    uint32x4_t sum0 = bar.add_mod4(va0, vb0);
                    uint32x4_t diff0 = bar.sub_mod4(va0, vb0);
                    uint32x4_t sum1 = bar.add_mod4(va1, vb1);
                    uint32x4_t diff1 = bar.sub_mod4(va1, vb1);
                    uint32x4_t sum2 = bar.add_mod4(va2, vb2);
                    uint32x4_t diff2 = bar.sub_mod4(va2, vb2);
                    uint32x4_t sum3 = bar.add_mod4(va3, vb3);
                    uint32x4_t diff3 = bar.sub_mod4(va3, vb3);
                    
                    // 乘以旋转因子
                    diff0 = bar.mul_mod4(diff0, w0);
                    diff1 = bar.mul_mod4(diff1, w1);
                    diff2 = bar.mul_mod4(diff2, w2);
                    diff3 = bar.mul_mod4(diff3, w3);
                    
                    // 存储结果
                    vst1q_u32((uint32_t*)(a+i+j), sum0);
                    vst1q_u32((uint32_t*)(a+i+j+mid), diff0);
                    vst1q_u32((uint32_t*)(a+i+j+W), sum1);
                    vst1q_u32((uint32_t*)(a+i+j+W+mid), diff1);
                    vst1q_u32((uint32_t*)(a+i+j+2*W), sum2);
                    vst1q_u32((uint32_t*)(a+i+j+2*W+mid), diff2);
                    vst1q_u32((uint32_t*)(a+i+j+3*W), sum3);
                    vst1q_u32((uint32_t*)(a+i+j+3*W+mid), diff3);
                }else {
              // 尾部标量
              for (int k=j; k<mid; ++k) {
                uint32_t x = a[i+k], y = a[i+k+mid];
                uint32_t sm = x+y>=p? x+y-p: x+y;
                uint32_t df = x>=y? x-y: x+p-y;
                df = br.mul_mod(df, wp[k]);
                a[i+k]=sm; a[i+k+mid]=df;
              }
            }
          }
        }
    }
}

// 逆变换 DIT-NTT（无初始、末尾位逆序）
void ntt_dit_barrett(int *a, int n, int p) {
    const int W = 4;
    BarrettNEON bar(p);
    BarrettReduction br(p);

    int lg = 0; while ((1<<lg) < n) lg++;
    for (int s = 1; s <= lg; ++s) {
        int len = 1<<s, mid = len>>1;
        // 正确的逆旋转因子： g^{(p-1) - (p-1)/len}
        int inv_wlen = fast_pow(g, (p-1) - (p-1)/len, p);
        std::vector<uint32_t> wp(mid);
        wp[0]=1;
        for (int j=1; j<mid; ++j) 
            wp[j] = br.mul_mod(wp[j-1], inv_wlen);

        for (int i=0; i<n; i+=len) {
          for (int j=0; j<mid; j+=W) {
            if (j+W<=mid) {
              uint32x4_t va = vld1q_u32((uint32_t*)(a+i+j));
              uint32x4_t vb = vld1q_u32((uint32_t*)(a+i+j+mid));
              uint32x4_t w4 = vld1q_u32(wp.data()+j);
              uint32x4_t vbw = bar.mul_mod4(vb, w4);
              uint32x4_t sum  = bar.add_mod4(va, vbw);
              uint32x4_t diff = bar.sub_mod4(va, vbw);
              vst1q_u32((uint32_t*)(a+i+j),      sum);
              vst1q_u32((uint32_t*)(a+i+j+mid), diff);
            } else {
              for (int k=j; k<mid; ++k) {
                uint32_t x = a[i+k], y = a[i+k+mid];
                uint32_t yw = br.mul_mod(y, wp[k]);
                uint32_t sm = x + yw >= p ? x+yw-p : x+yw;
                uint32_t df = x >= yw ? x-yw : x+p-yw;
                a[i+k]=sm; a[i+k+mid]=df;
              }
            }
          }
        }
    }
    // 乘 n^{-1}
    int ninv = fast_pow(n, p-2, p);
    for (int i=0; i<n; ++i)
      a[i] = br.mul_mod(a[i], ninv);
}
// 使用 Barrett 归约的多项式乘法
void poly_multiply_optimizedB(int *a, int *b, int *ab, int n, int p) {
    int m = 1;
    while (m < 2*n-1) m <<= 1;  // 找到≥2n-1的最小2次幂
    
    // 复制输入数据
    int *a_copy = new int[m]();
    int *b_copy = new int[m]();
    
    memcpy(a_copy, a, n * sizeof(int));
    memcpy(b_copy, b, n * sizeof(int));
    
    // 使用 Barrett 归约的 NTT
    ntt_dif_barrett(a_copy, m, p);
    ntt_dif_barrett(b_copy, m, p);
    
    // 点乘
    BarrettReduction br(p);
    for (int i = 0; i < m; i++) {
        a_copy[i] = br.mul_mod(a_copy[i], b_copy[i]);
    }
    
    // 逆 NTT
    ntt_dit_barrett(a_copy, m, p);
    // 复制结果
    for (int i = 0; i < 2*n-1; i++) {
        ab[i] = ((a_copy[i] % p) + p) % p; // 确保结果在[0,p)范围内
    }
    delete[] a_copy;
    delete[] b_copy;
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
        poly_multiply_optimizedB(a,b,ab,n_,p_);
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