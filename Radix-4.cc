#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include<bits/stdc++.h>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <arm_neon.h>
typedef long long int ll;
const int g = 3;
// 可以自行添加需要的头文件
alignas(32) int rev[30000000]={0};
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
    inline void butterfly4_dif(uint32x4_t &x0, uint32x4_t &x1, uint32x4_t &x2, uint32x4_t &x3,uint32x4_t w1, uint32x4_t w2, uint32x4_t w3, uint32x4_t imag) const {
            uint32x4_t t0 = vaddq_u32(x0, x2);
            t0 = normalize4(t0);

            uint32x4_t t1 = vsubq_u32(vaddq_u32(x0, vm), x2);
            t1 = normalize4(t1);

            uint32x4_t t2 = vaddq_u32(x1, x3);
            t2 = normalize4(t2);

            uint32x4_t t3 = vsubq_u32(vaddq_u32(x1, vm), x3);
            t3 = normalize4(t3);

            // 第2步：应用虚数单位j到t3
            t3 = mul4(t3, imag);
            t3 = normalize4(t3);

            // 第3步：计算最终输出
            uint32x4_t y0 = vaddq_u32(t0, t2);
            y0 = normalize4(y0);

            uint32x4_t y1 = vaddq_u32(t1, t3);
            y1 = normalize4(y1);

            uint32x4_t y2 = vsubq_u32(vaddq_u32(t0, vm), t2);
            y2 = normalize4(y2);

            uint32x4_t y3 = vsubq_u32(vaddq_u32(t1, vm), t3);
            y3 = normalize4(y3);

            // 第4步：应用旋转因子
            x0 = y0;  // w0 = 1
            x1 = mul4(y1, w1);
            x1 = normalize4(x1);

            x2 = mul4(y2, w2);
            x2 = normalize4(x2);

            x3 = mul4(y3, w3);
            x3 = normalize4(x3);
            }
};

typedef long long int ll;

// 辅助函数：快速幂取模
int pow_mod(int a, int b, int p) {
    int res = 1;
    while (b) {
        if (b & 1) res = (ll)res * a % p;
        a = (ll)a * a % p;
        b >>= 1;
    }
    return res;
}

// 辅助函数：模逆元
int mod_inverse(int a, int p) {
    return pow_mod(a, p - 2, p);
}

// 生成四进制的位逆序表
void generate_rev(int *rev, int n) {
    int log_n = 0;
    while ((1 << (2 * log_n)) < n) log_n++;
    for (int i = 0; i < n; ++i) {
        int x = i;
        int reversed = 0;
        for (int j = 0; j < log_n; ++j) {
            reversed = (reversed << 2) | (x & 3);
            x >>= 2;
        }
        rev[i] = reversed;
    }
}

// Radix-4 DIF NTT
void ntt_radix4(int *a, int n, int p, int g) {
    int m = 0;
    while ((1 << (2 * m)) < n) m++;
    
    generate_rev(rev, n);
    
    int i = pow_mod(g, (p - 1) / 4, p); // 四次单位根
    
    for (int s = m; s >= 1; --s) {
        int len = 1 << (2 * s);
        int len_block = len >> 2;
        int w_base = pow_mod(g, (p - 1) / len, p);
        
        for (int k = 0; k < n; k += len) {
            for (int j = 0; j < len_block; ++j) {
                int w_j = pow_mod(w_base, j, p);
                int w2j = (ll)w_j * w_j % p;
                int w3j = (ll)w_j * w2j % p;
                
                int x0 = a[k + j];
                int x1 = a[k + j + len_block];
                int x2 = a[k + j + 2 * len_block];
                int x3 = a[k + j + 3 * len_block];
                
                // 蝶形运算
                ll t0 = ((ll)x0 + x1 + x2 + x3) % p;
                ll t1 = ((ll)x0 + (ll)i * x1 - x2 - (ll)i * x3) % p * w_j % p;
                ll t2 = ((ll)x0 - x1 + x2 - x3) % p * w2j % p;
                ll t3 = ((ll)x0 - (ll)i * x1 - x2 + (ll)i * x3) % p * w3j % p;
                
                a[k + j] = t0;
                a[k + j + len_block] = (t1 + p) % p;
                a[k + j + 2 * len_block] = (t2 + p) % p;
                a[k + j + 3 * len_block] = (t3 + p) % p;
            }
        }
    }
    
    // 位逆序置换
    int *tmp = new int[n];
    memcpy(tmp, a, sizeof(int) * n);
    for (int i = 0; i < n; ++i) a[rev[i]] = tmp[i];
    delete[] tmp;
}

// 逆NTT
void intt_radix4(int *a, int n, int p, int g) {
    int inv_g = mod_inverse(g, p);
    ntt_radix4(a, n, p, inv_g);
    int inv_n = mod_inverse(n, p);
    for (int i = 0; i < n; ++i) a[i] = (ll)a[i] * inv_n % p;
}

// 多项式乘法
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    int m = 1;
    while (m < 2 * n) m <<= 2; // 扩展到4的幂
    if (m < 4) m = 4;
    
    int *a_ext = new int[m]();
    int *b_ext = new int[m]();
    memcpy(a_ext, a, sizeof(int) * n);
    memcpy(b_ext, b, sizeof(int) * n);
    
    // 正向NTT
    ntt_radix4(a_ext, m, p, g);
    ntt_radix4(b_ext, m, p, g);
    
    // 逐点相乘
    for (int i = 0; i < m; ++i)
        ab[i] = (ll)a_ext[i] * b_ext[i] % p;
    
    // 逆NTT
    intt_radix4(ab, m, p, g);
    
    // 截断结果
    for (int i = 2 * n - 1; i < m; ++i) ab[i] = 0;
    delete[] a_ext;
    delete[] b_ext;
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
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply_neon(a,b,ab,n_,p_);
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