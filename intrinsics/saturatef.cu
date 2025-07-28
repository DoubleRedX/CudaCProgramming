//
// Created by Gary27 on 2025/6/25.
//

#include <cstdio>

__global__ void saturateDemo() {
    // float vals[] = {
    //     -0.5f, 0.2f, 0.8f, 1.5f,
    //     -0.010101f, 0.22313f
    // };
    // for (auto e: vals) {
    //     float x = e;
    //     float y = __saturatef(x);
    //     printf("x = %f -> __saturatef(x) = %f\n", x, y);
    // }

    int CR_BIN_WARPS = 16;
    unsigned int thrInBlock = threadIdx.x + threadIdx.y * 32;
    bool act = (thrInBlock < CR_BIN_WARPS);
    unsigned int actMask = __ballot_sync(~0u, act);
    if (act == true) {
        printf("%d\n", actMask);
    }

}

int main(){
    saturateDemo<<<dim3(1, 1, 1), dim3(32, 4)>>>();
    cudaDeviceSynchronize();

return 0;
}