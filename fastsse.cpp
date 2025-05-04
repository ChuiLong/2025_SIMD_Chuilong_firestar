#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <sys/time.h>
#include <algorithm>
#include <emmintrin.h>
#include <immintrin.h>

#include <vector>
typedef long long int ll;
#define SSE_ALIGNMENT 16
#define CACHE_LINE_SIZE 64
const int g = 3;

ll rev[300000] __attribute__((aligned(16))) = {0};
int a[300000] __attribute__((aligned(16)));
int b[300000] __attribute__((aligned(16)));
int ab[300000] __attribute__((aligned(16)));

struct BarrettReduction {
    uint32_t m;
    uint32_t k;
    uint64_t mu;

    BarrettReduction(uint32_t m_) : m(m_) {
        assert(m != 0);
        k = 32 - __builtin_clz(m - 1);
        mu = (__uint128_t(1) << (k + k)) / m;
    }

    uint32_t reduce(uint64_t a) const {
        if (a < m) return a;
        __uint128_t product = (__uint128_t)a * mu;
        uint64_t q = product >> (k + k);
        uint32_t r = a - q * m;
        return r >= m ? r - m : r;
    }

    uint32_t mul_mod(uint32_t a, uint32_t b) const {
        return reduce((__uint128_t)a * b);
    }
};

struct BarrettSSE {
    const BarrettReduction br;
    const __m128i vp;
    const __m128i vm;

    BarrettSSE(uint32_t p) : br(p), vp(_mm_set1_epi32(p)), 
                            vm(_mm_set1_epi32(p-1)) {}

    __m128i mul_mod4(__m128i a, __m128i b) const {
        // SSE4.1优化版本
        __m128i product = _mm_mul_epu32(a, b);
        __m128i q = _mm_srli_epi64(product, br.k);
        __m128i qm = _mm_mul_epu32(q, vp);
        __m128i r = _mm_sub_epi32(a, qm);
        return _mm_add_epi32(r, _mm_and_si128(
            _mm_cmplt_epi32(r, _mm_setzero_si128()), vp));
    }

    __m128i add_mod4(__m128i a, __m128i b) const {
        __m128i sum = _mm_add_epi32(a, b);
        __m128i cmp = _mm_cmpgt_epi32(sum, vm);
        return _mm_sub_epi32(sum, _mm_and_si128(cmp, vp));
    }

    __m128i sub_mod4(__m128i a, __m128i b) const {
        __m128i diff = _mm_sub_epi32(a, b);
        return _mm_add_epi32(diff, _mm_and_si128(
            _mm_cmplt_epi32(diff, _mm_setzero_si128()), vp));
    }
};

int fast_pow(int x, int y, int p) {
    int res = 1;
    x %= p;
    while (y > 0) {
        if (y & 1) res = (1LL * res * x) % p;
        x = (1LL * x * x) % p;
        y >>= 1;
    }
    return res;
}

void ntt_dif_sse(int *a, int n, int p) {
    BarrettSSE bar(p);
    int lg = 32 - __builtin_clz(n-1); // 优化log2计算

    {
        // 预分配线程本地存储
        alignas(SSE_ALIGNMENT) uint32_t w_tmp[4];

        for (int s = lg; s >= 1; --s) {
            int len = 1 << s, mid = len >> 1;
            int wlen = fast_pow(g, (p-1)/len, p);

            // 预生成旋转因子
            alignas(SSE_ALIGNMENT) uint32_t wp[mid];
            wp[0] = 1;
            for(int j=1; j<mid; ++j)
                wp[j] = bar.br.mul_mod(wp[j-1], wlen);

            for (int i = 0; i < n; i += len) {
                // 预取下个缓存块
                if(i + len < n) 
                    _mm_prefetch((char*)(a+i+len), _MM_HINT_T0);

                for (int j=0; j<mid; j+=4) {
                    // 向量化加载旋转因子
                    memcpy(w_tmp, &wp[j], 16);
                    __m128i vw = _mm_load_si128((__m128i*)w_tmp);

                    // 合并加载和计算
                    __m128i* ptr1 = (__m128i*)(a + i + j);
                    __m128i* ptr2 = (__m128i*)(a + i + j + mid);
                    __m128i va = _mm_load_si128(ptr1);
                    __m128i vb = _mm_load_si128(ptr2);

                    __m128i sum = bar.add_mod4(va, vb);
                    __m128i diff = bar.sub_mod4(va, vb);
                    diff = bar.mul_mod4(diff, vw);

                    _mm_store_si128(ptr1, sum);
                    _mm_stream_si128(ptr2, diff); // 非临时存储
                }
            }
        }
    }
}

void ntt_dit_sse(int *a, int n, int p) {
    BarrettSSE bar(p);
    BarrettReduction br(p);
    int lg = 0;
    while ((1 << lg) < n) lg++;

    for (int s = 1; s <= lg; ++s) {
        int len = 1 << s, mid = len >> 1;
        int inv_wlen = fast_pow(g, (p-1) - (p-1)/len, p);

        std::vector<uint32_t> wp(mid);
        wp[0] = 1;
        for (int j = 1; j < mid; ++j)
            wp[j] = br.mul_mod(wp[j-1], inv_wlen);

        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < mid; j += 4) {
                if (j + 4 <= mid) {
                    __m128i va = _mm_load_si128((__m128i*)(a + i + j));
                    __m128i vb = _mm_load_si128((__m128i*)(a + i + j + mid));

                    alignas(16) uint32_t w_tmp[4] = {wp[j], wp[j+1], wp[j+2], wp[j+3]};
                    __m128i vw = _mm_load_si128((__m128i*)w_tmp);

                    __m128i vbw = bar.mul_mod4(vb, vw);
                    __m128i sum = bar.add_mod4(va, vbw);
                    __m128i diff = bar.sub_mod4(va, vbw);

                    _mm_store_si128((__m128i*)(a + i + j), sum);
                    _mm_store_si128((__m128i*)(a + i + j + mid), diff);
                } else {
                    for (int k = j; k < mid; ++k) {
                        uint32_t x = a[i + k], y = a[i + k + mid];
                        uint32_t yw = br.mul_mod(y, wp[k]);
                        uint32_t sm = (x + yw) % p;
                        uint32_t df = (x - yw + p) % p;
                        a[i + k] = sm;
                        a[i + k + mid] = df;
                    }
                }
            }
        }
    }

    int ninv = fast_pow(n, p-2, p);
    for (int i = 0; i < n; ++i)
        a[i] = br.mul_mod(a[i], ninv);
}

void poly_multiply_sse(int *a, int *b, int *ab, int n, int p) {
    int m = 1 << (32 - __builtin_clz(2*n-2)); // 优化长度计算

    // 使用批量内存初始化
    int *a_copy = (int*)_mm_malloc(m * sizeof(int), SSE_ALIGNMENT);
    int *b_copy = (int*)_mm_malloc(m * sizeof(int), SSE_ALIGNMENT);
    memset(a_copy, 0, m * sizeof(int));
    memset(b_copy, 0, m * sizeof(int));
    memcpy(a_copy, a, n * sizeof(int));
    memcpy(b_copy, b, n * sizeof(int));

    // 批量处理点乘
    for(int i=0; i<m; i+=4) {
        __m128i va = _mm_load_si128((__m128i*)(a_copy + i));
        __m128i vb = _mm_load_si128((__m128i*)(b_copy + i));
        _mm_store_si128((__m128i*)(a_copy + i), 
                       BarrettSSE(p).mul_mod4(va, vb));
    }

    ntt_dit_sse(a_copy, m, p);

    // 最后结果处理优化
    for(int i=0; i<2*n-1; ++i)
        ab[i] = (a_copy[i] % p + p) % p;

    _mm_free(a_copy);
    _mm_free(b_copy);
}


// 浠ヤ笅涓哄師鏈夋鏋朵唬鐮侊紝淇濇寔涓嶅彉
void fRead(int *a, int *b, int *n, int *p, int input_id){

    std::string str1 = "C:/Users/user/Downloads/nttdata/";
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
    std::string str1 = "C:/Users/user/Downloads/nttdata/";
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
            std::cout<<"澶氶」寮忎箻娉曠粨鏋滈敊璇?"<<std::endl;
            return;
        }
    }
    std::cout<<"澶氶」寮忎箻娉曠粨鏋滄纭?"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    std::string str1 = "C:/Users/86180/Downloads/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + "true.out";
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

int main()
{
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab,0,sizeof(ab));
        memset(rev,0,sizeof(rev));
        auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply_sse(a,b,ab,n_,p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}