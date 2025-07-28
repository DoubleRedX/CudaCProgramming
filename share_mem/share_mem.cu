//
// Created by Gary27 on 2025/6/26.
//
#include <cstdio>
#define CR_BIN_WARPS 16

__global__ void share_mem_write() {
    __shared__ volatile unsigned int s_broadcast [CR_BIN_WARPS + 16];

    // auto tid = threadIdx.x + threadIdx.y * 32;

    int myIdx = 0;
    int num = 1;

    if (threadIdx.x == 31){ // Do not assume that last thread in warp wins the write.
        s_broadcast[threadIdx.y + 16] = myIdx + num;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {

        printf("arr 0 = %d\n", s_broadcast[0]);
        printf("arr 1 = %d\n", s_broadcast[1]);
        printf("arr 2 = %d\n", s_broadcast[2]);
        printf("arr 3 = %d\n", s_broadcast[3]);
        printf("arr 4 = %d\n", s_broadcast[4]);
        printf("arr 5 = %d\n", s_broadcast[5]);
        printf("arr 6 = %d\n", s_broadcast[6]);
        printf("arr 7 = %d\n", s_broadcast[7]);
        printf("arr 8 = %d\n", s_broadcast[8]);
        printf("arr 9 = %d\n", s_broadcast[9]);
        printf("arr 10 = %d\n", s_broadcast[10]);
        printf("arr 11 = %d\n", s_broadcast[11]);
        printf("arr 12 = %d\n", s_broadcast[12]);
        printf("arr 13 = %d\n", s_broadcast[13]);
        printf("arr 14 = %d\n", s_broadcast[14]);
        printf("arr 15 = %d\n", s_broadcast[15]);
        printf("arr 16 = %d\n", s_broadcast[16]);
        printf("arr 17 = %d\n", s_broadcast[17]);
        printf("arr 18 = %d\n", s_broadcast[18]);
        printf("arr 19 = %d\n", s_broadcast[19]);
        printf("arr 20 = %d\n", s_broadcast[20]);
        printf("arr 21 = %d\n", s_broadcast[21]);
        printf("arr 22 = %d\n", s_broadcast[22]);
        printf("arr 23 = %d\n", s_broadcast[23]);
        printf("arr 24 = %d\n", s_broadcast[24]);
        printf("arr 25 = %d\n", s_broadcast[25]);
        printf("arr 26 = %d\n", s_broadcast[26]);
        printf("arr 27 = %d\n", s_broadcast[27]);
        printf("arr 28 = %d\n", s_broadcast[28]);
        printf("arr 29 = %d\n", s_broadcast[29]);
        printf("arr 30 = %d\n", s_broadcast[30]);
        printf("arr 31 = %d\n", s_broadcast[31]);

    }

}


int main(){
    share_mem_write<<<dim3(1,1,1), dim3(32, 2, 1)>>>();
    cudaDeviceSynchronize();



    return 0;
}