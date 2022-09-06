#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cassert>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <arm_neon.h>

// utility string function
std::vector<std::string> split (std::string s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (getline (ss, item, delim)) {
        result.push_back (item);
    }
    return result;
}

// custom image class to read and write ascii pgm image
template <typename T>
struct Image{
    int height;
    int width;
    std::vector<T> data;
    Image() = default;
    Image(int height, int width){
        this->height=height;
        this->width=width;
        data = std::vector<T>(height*width,0);
    }
    bool read(std::string filename){
        std::string buffer;
        std::ifstream f(filename);
        std::getline(f,buffer);
        assert(buffer=="P2" && "Wrong file format");
        std::getline(f,buffer);
        std::vector<std::string> splitted = split(buffer,' ');
        assert(splitted.size()==2 && "Wrong image size entry. Shape dimension not equal (2,)");
        height = std::stoi(splitted[0]);
        width = std::stoi(splitted[1]);
        std::cout<<"Loading image of shape "<<height<<"[Height] x "<<width<<"[Width]"<<std::endl;
        std::getline(f,buffer); //discard `MaxVal` line
        for (std::string line; std::getline(f, line); )
        {
            splitted = split(line,' ');
            for (const auto& val : splitted){
                if(val.find_first_not_of(' ') != std::string::npos) {
                    int parsedInt = std::stoi(val);
                    data.push_back((T) parsedInt);
                }
            }
        }
        std::cout<<"Data loaded ["<<data.size()<<" bytes]"<<std::endl;
        if (data.size()==width*height){
            return true;
        }
        else{
            return false;
        }
    }
    bool write(std::string filename){
        std::ofstream f(filename);
        f << "P2\n";
        f << height << "  " << width << "\n";
        f << "255\n";
        for (int r = 0; r < height; r++) {
            std::stringstream ss;
            ss << std::to_string(data[r * height]);
            for (int c = 1; c < width; c++) {
                ss << "  " << std::to_string(data[r * height + c]);
            }
            f << ss.str() << std::endl;
        }
        return true;
    }
};

float ssd_reduce(const float* ptrA, const float* ptrB, uint32_t count);

float ssd_reduce_K3(const float* ptrA, const float* ptrB);

void nlm_unoptimized(int N,
                     int K,
                     float h,
                     int padLen,
                     int threadCount,
                     const Image<float> &padded,
                     Image<float> &output,
                     Image<float> &B);

template <bool fixedK>
void nlm_neon(int N,
              int K,
              float h,
              int padLen,
              int threadCount,
              const Image<float> &padded,
              Image<float> &output,
              Image<float> &B);

int main(int argc, char** argv) {
    int option = 0;
    int threadCount=1;
    if(argc>=2){
        int parsed = std::stoi(argv[1]);
        switch (parsed) {
            case 2:
                option = 2;
                std::cout << "Neon optimized v2 - Fixed K" << std::endl;
                break;
            case 1:
                option = 1;
                std::cout << "Neon optimized v1" << std::endl;
                break;
            default:
                std::cout << "Unoptimized version" << std::endl;
                break;
        }
    }
    if(argc==3){
        threadCount = std::thread::hardware_concurrency();
    }

    // Read input
    Image<uint8_t> im_byte;
    if(im_byte.read("./lena_128x128_sigma_0.2.ascii.pgm")){
        std::cout<<"Image loaded"<<std::endl;
    }
    else{
        std::cout<<"Failed to load image"<<std::endl;
    }
    Image<float> im(im_byte.height,im_byte.width);
    std::transform(im_byte.data.begin(),
                   im_byte.data.end(),
                   im.data.begin(),
                   [](auto i){ return i/255.0; }
                  );

    auto start = std::chrono::high_resolution_clock::now();
    // TODO
    int N=9;
    int K=3;
    float h=0.2;
    int padLen= N+K;
    Image<float> output(im.height,im.width);
    Image<float> C(im.height,im.width);
    Image<float> padded(im.height + 2*padLen,im.width + 2*padLen);
    // fill in padded
    for (int y = 0; y < im.height; y++) {
        for (int x = 0; x < im.width; x++) {
            padded.data[(y+padLen)+(x+padLen)*padded.height] = im.data[y+x*im.height];
        }
    }
    // nlm denoising
    std::vector<float> inputA(7,1.0);
    std::vector<float> inputB(7,3.0);
    float ret;
    switch (option) {
        case 2:
            std::cout<<"Running nlm_neon_fixed_k:"<<std::endl;
            nlm_neon<true>(N, K, h, padLen, threadCount, padded, output, C);
            break;
        case 1:
            std::cout<<"Running nlm_neon:"<<std::endl;
            nlm_neon<false>(N, K, h, padLen, threadCount, padded, output, C);
            break;
        case 0:
        default:
            std::cout<<"Running nlm_unoptimized:"<<std::endl;
            nlm_unoptimized(N, K, h, padLen,threadCount,  padded, output, C);
            break;
    }

    Image<uint8_t> result(im.height,im.width);
    std::transform (
            output.data.begin(),
            output.data.end(),
            C.data.begin(),
            result.data.begin(),
            [](auto i, auto j){
                return std::floor(i/j*255.0);
            });

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = 1e-6 * (end-start).count();
    std::cout<<"Execution time ="<< elapsed_ms << " milliseconds\n";

    // Write output
    result.write("./denoised.ascii.pgm");

    return 0;
}

void nlm_unoptimized(int N,
                     int K,
                     float h,
                     int padLen,
                     int threadCount,
                     const Image<float> &padded,
                     Image<float> &output,
                     Image<float> &C) {
    int kernelArea = (2*K+1)*(2*K+1);
    float hSquared = h*h;
    std::cout<<"Thread count ["<<threadCount<<"]\n";
#pragma omp parallel for num_threads(threadCount)
    for (int y = 0; y < output.height; y++) {
        for (int x = 0; x < output.width; x++) {
            for (int ny = -N; ny < N+1; ny++) {
                for (int nx = -N; nx < N+1; nx++) {
                    float ssd = 0.f;
                    for (int ky = -K; ky < K + 1; ky++) {
                        for (int kx = -K; kx < K + 1; kx++) {
                            float diff;
                            diff = padded.data[(padLen + y + ny + ky) + (padLen + x + nx + kx) * padded.height] -
                                    padded.data[(padLen + y + ky) + (padLen + x + kx) * padded.height];
                            ssd += (diff * diff);
                        }
                    }
                    float dSquared = ssd/kernelArea;
                    float ex = std::exp(-dSquared/hSquared);
                    C.data[y + x*C.height] += ex;
                    output.data[y + x*output.height] += ex*padded.data[(padLen+y+ny) + (padLen+x+nx)*padded.height];
                }
            }
        }
    }
}

template <bool fixedK>
void nlm_neon(int N,
              int K,
              float h,
              int padLen,
              int threadCount,
              const Image<float> &padded,
              Image<float> &output,
              Image<float> &C){
    int kernelWidth = 2*K+1;
    int kernelArea = kernelWidth*kernelWidth;
    float hSquared = h*h;
    std::cout<<"Thread count ["<<threadCount<<"]\n";
#pragma omp parallel for num_threads(threadCount)
    for (int y = 0; y < output.height; y++) {
        for (int x = 0; x < output.width; x++) {
            for (int ny = -N; ny < N+1; ny++) {
                for (int nx = -N; nx < N+1; nx++) {
                    float ssd = 0.f;
                    // for kernel convolution, we iterate along the x-axis only,
                    // at each iteration, process (2*K)+1 contiguous elements along y-axis
                    for (int kx = -K; kx < K + 1; kx++) {
                        int refIndex = (padLen + y + ny) + (padLen + x + nx + kx) * padded.height;
                        int kernelIndex = (padLen + y) + (padLen + x + kx) * padded.height;
                        if (fixedK){
                            ssd += ssd_reduce_K3(padded.data.data()+kernelIndex, padded.data.data()+refIndex);
                        }
                        else{
                            ssd += ssd_reduce(padded.data.data()+kernelIndex, padded.data.data()+refIndex, kernelWidth);
                        }

                    }
                    float dSquared = ssd/kernelArea;
                    float ex = std::exp(-dSquared/hSquared);
                    C.data[y + x*C.height] += ex;
                    output.data[y + x*output.height] += ex*padded.data[(padLen+y+ny) + (padLen+x+nx)*padded.height];
                }
            }
        }
    }
}

const int SIMD_MULTPLE = 4;

// neon simd utility function
float ssd_reduce(const float* ptrA, const float* ptrB, uint32_t count) {
    int remainder = count % SIMD_MULTPLE;
    int fullLoopCount = count/SIMD_MULTPLE; //floor
    int fullLoopEnd = (fullLoopCount-1)*SIMD_MULTPLE ;

    float32x2_t vec64a, vec64b;
    float32x4_t vec128 = vdupq_n_f32(0.0); // clear accumulators
    float32x4_t vecA, vecB;

    // full stride, contiguous memory access loop
    for (int i = 0; i <= fullLoopEnd; i+=SIMD_MULTPLE) {
        vecA = vld1q_f32(ptrA+i); // load four 32-bit values
        vecB = vld1q_f32(ptrB+i); // load four 32-bit values
        float32x4_t diff = vsubq_f32(vecA,vecB);
        float32x4_t squared = vmulq_f32(diff,diff);
        vec128=vaddq_f32(vec128, squared); // accumulate the squared_diff
    }
    // remainder loop
    if(remainder != 0){
        int remainderFirstElement = (fullLoopCount)*SIMD_MULTPLE;
        vecA = vld1q_f32(ptrA+remainderFirstElement); // load four 32-bit values
        vecB = vld1q_f32(ptrB+remainderFirstElement); // load four 32-bit values

        // set remainder to 0
        switch (SIMD_MULTPLE - remainder) {
            case 3:
                vsetq_lane_f32(0,vecA,1);
                vsetq_lane_f32(0,vecB,1);
            case 2:
                vsetq_lane_f32(0,vecA,2);
                vsetq_lane_f32(0,vecB,2);
            case 1:
            default:
                vsetq_lane_f32(0,vecA,3);
                vsetq_lane_f32(0,vecB,3);
                break;
        }

        float32x4_t diff = vsubq_f32(vecA,vecB);
        float32x4_t squared = vmulq_f32(diff,diff);
        vec128=vaddq_f32(vec128, squared); // accumulate the squared_diff

    }
    vec64a = vget_low_f32(vec128); // split 128-bit vector

    vec64b = vget_high_f32(vec128); // into two 64-bit vectors

    vec64a = vadd_f32 (vec64a, vec64b); // add 64-bit vectors together

    float result = vget_lane_f32(vec64a, 0); // extract lanes and

    result += vget_lane_f32(vec64a, 1); // add together scalars

    return result;
}

const int K3_COUNT = 7; // K=3; count=2*K+1
const int K3_FULL_LOOP_COUNT = K3_COUNT/SIMD_MULTPLE; //floor
const int K3_FULL_LOOP_END = (K3_FULL_LOOP_COUNT-1)*SIMD_MULTPLE;
const int K3_REMAINDER_LOOP_START = (K3_FULL_LOOP_COUNT)*SIMD_MULTPLE;

float ssd_reduce_K3(const float* ptrA, const float* ptrB) {
    float32x2_t vec64a, vec64b;
    float32x4_t vec128 = vdupq_n_f32(0.0); // clear accumulators
    float32x4_t vecA, vecB;

    // full stride, contiguous memory access loop
    for (int i = 0; i <= K3_FULL_LOOP_END; i+=SIMD_MULTPLE) {
        vecA = vld1q_f32(ptrA+i); // load four 32-bit values
        vecB = vld1q_f32(ptrB+i); // load four 32-bit values
        float32x4_t diff = vsubq_f32(vecA,vecB);
        vec128 = vmlaq_f32(vec128, diff, diff); // multiply-accumulate the diff
    }

    // Load remainder loop
    vecA = vld1q_f32(ptrA+K3_REMAINDER_LOOP_START); // load four 32-bit values
    vecB = vld1q_f32(ptrB+K3_REMAINDER_LOOP_START); // load four 32-bit values

    // remainder = K3_COUNT % SIMD_MULTPLE = 7 % 4 = 3
    // SIMD_MULTPLE - remainder = 4 - 3 = 1
    // we need to set the last lane to 0
    vsetq_lane_f32(0,vecA,3);
    vsetq_lane_f32(0,vecB,3);

    float32x4_t diff = vsubq_f32(vecA,vecB);
    vec128 = vmlaq_f32(vec128, diff, diff); // multiply-accumulate the diff

    vec64a = vget_low_f32(vec128); // split 128-bit vector

    vec64b = vget_high_f32(vec128); // into two 64-bit vectors

    vec64a = vadd_f32 (vec64a, vec64b); // add 64-bit vectors together

    float result = vget_lane_f32(vec64a, 0); // extract lanes and

    result += vget_lane_f32(vec64a, 1); // add together scalars

    return result;
}

template void nlm_neon<true>(
        int N,
        int K,
        float h,
        int padLen,
        int threadCount,
        const Image<float> &padded,
        Image<float> &output,
        Image<float> &C);

template void nlm_neon<false>(
        int N,
        int K,
        float h,
        int padLen,
        int threadCount,
        const Image<float> &padded,
        Image<float> &output,
        Image<float> &C);