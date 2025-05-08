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
const int g = 3;


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
    __m128i m4, k4; 
    
    BarrettSSE(uint32_t p) : br(p) {
        m4 = _mm_set1_epi32(p);
        k4 = _mm_set1_epi32(br.k);
    }
    
    __m128i add_mod4(__m128i a, __m128i b) const {
        __m128i sum = _mm_add_epi32(a, b);
        __m128i cmp = _mm_cmpgt_epi32(sum, m4);
        __m128i mask = _mm_and_si128(m4, cmp);
        return _mm_sub_epi32(sum, mask);
    }
    
    __m128i sub_mod4(__m128i a, __m128i b) const {
        __m128i diff = _mm_sub_epi32(a, b);
        __m128i cmp = _mm_cmplt_epi32(diff, _mm_setzero_si128());
        __m128i mask = _mm_and_si128(m4, cmp);
        return _mm_add_epi32(diff, mask);
    }

    __m128i mul_mod4(__m128i a, __m128i b) const {
        uint32_t a0 = _mm_extract_epi32(a, 0);
        uint32_t a1 = _mm_extract_epi32(a, 1);
        uint32_t a2 = _mm_extract_epi32(a, 2);
        uint32_t a3 = _mm_extract_epi32(a, 3);

        uint32_t b0 = _mm_extract_epi32(b, 0);
        uint32_t b1 = _mm_extract_epi32(b, 1);
        uint32_t b2 = _mm_extract_epi32(b, 2);
        uint32_t b3 = _mm_extract_epi32(b, 3);

        return _mm_setr_epi32(
            br.mul_mod(a0, b0),
            br.mul_mod(a1, b1),
            br.mul_mod(a2, b2),
            br.mul_mod(a3, b3)
        );
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
    BarrettReduction br(p);
    int lg = 0;
    while ((1 << lg) < n) lg++;

    for (int s = lg; s >= 1; --s) {
        int len = 1 << s, mid = len >> 1;
        int wlen = fast_pow(g, (p-1)/len, p);

        uint32_t wp[mid];
        wp[0] = 1;
        for (int j = 1; j < mid; ++j)
            wp[j] = br.mul_mod(wp[j-1], wlen);

        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < mid; j += 4) {
                if (j + 4 <= mid) {
                    __m128i va = _mm_load_si128((__m128i*)(a + i + j));
                    __m128i vb = _mm_load_si128((__m128i*)(a + i + j + mid));
                    
                    alignas(16) uint32_t w_tmp[4] = {wp[j], wp[j+1], wp[j+2], wp[j+3]};
                    __m128i vw = _mm_load_si128((__m128i*)w_tmp);

                    __m128i sum = bar.add_mod4(va, vb);
                    __m128i diff = bar.sub_mod4(va, vb);
                    diff = bar.mul_mod4(diff, vw);

                    _mm_store_si128((__m128i*)(a + i + j), sum);
                    _mm_store_si128((__m128i*)(a + i + j + mid), diff);
                } else {
                    for (int k = j; k < mid; ++k) {
                        uint32_t x = a[i + k], y = a[i + k + mid];
                        uint32_t sm = (x + y) % p;
                        uint32_t df = br.mul_mod((x - y + p) % p, wp[k]);
                        a[i + k] = sm;
                        a[i + k + mid] = df;
                    }
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

        uint32_t wp[mid];
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
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;
    std::fill(a+n, a+m, 0);
    std::fill(b+n, b+m, 0);
    ntt_dif_sse(a, m, p);
    ntt_dif_sse(b, m, p);
    BarrettSSE bar(p);
    for (int i = 0; i < m; i += 4) {
        __m128i va = _mm_load_si128((__m128i*)(a + i));
        __m128i vb = _mm_load_si128((__m128i*)(b+ i));
        __m128i vab = bar.mul_mod4(va, vb);
        _mm_store_si128((__m128i*)(a + i), vab);
    }

    ntt_dit_sse(a, m, p);

    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = (a[i] % p + p) % p;

}

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
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
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
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    std::string str1 = "C:/Users/user/Downloads/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + "true.out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
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
