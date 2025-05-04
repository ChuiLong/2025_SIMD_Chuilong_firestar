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
typedef long long int ll;
const int g = 3;
// 可以自行添加需要的头文件

ll rev[300000] __attribute__((aligned(16))) = {0};
int a[300000] __attribute__((aligned(16)));
int b[300000] __attribute__((aligned(16)));
int ab[300000] __attribute__((aligned(16)));
void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
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
    // 判断多项式乘法结果是否正确
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
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "C:/Users/86180/Downloads/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + "slow.out";
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
static inline int mod_pow(int x, int y, int p) {
    ll z = 1ll * x, ans = 1;
    while (y) {
        if (y & 1) ans = ans * z % p;
        z = z * z % p;
        y >>= 1;
    }
    return (int)ans;
}

void ntt(int *a, int n, int p,int inv){
    int startbit = 0;
    while((1<<startbit)<n)startbit++;

    for(int i = 0;i<n;i++){
        rev[i] = (rev[i>>1])>>1 | ((i&1)<<(startbit-1));
        if(i<rev[i]) std::swap(a[i],a[rev[i]]);
    }

    for(int mid = 1;mid<n;mid*=2){
        int tmp = pow(g,(p-1)/(mid*2),p);
        if(inv == -1) tmp=pow(tmp,p-2,p);
        for(int i = 0;i<n;i+=mid*2){
            ll omega = 1;
            for(int j = 0;j<mid;j++,omega = omega*tmp%p){
                ll x = a[i+j],y=omega*a[i+j+mid]%p;
                a[i+j] = (x+y)%p,a[i+j+mid]=(x-y+p)%p;
            }
        }
    }
}

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    int m = 1;
while (m < 2*n-1) m <<= 1;         // 找到 ≥2n-1 的最小二次幂
// 把 a, b 扩展到长度 m
std::fill(a+n, a+m, 0);
std::fill(b+n, b+m, 0);
// 然后用 m 调用 ntt/INTT
    ntt(a, m, p,1);  
    ntt(b, m,p,1); 

    for (int i = 0; i < m; ++i) {
        ab[i] = (1ll*a[i] * b[i]) % p;
    }
    ntt(ab, m, p,-1); 
    ll n_inv = pow(m, p - 2, p); 
    for (int i = 0; i < m; ++i) {
        ab[i] = (1LL * ab[i] * n_inv) % p;
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
        poly_multiply(a,b,ab,n_,p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}