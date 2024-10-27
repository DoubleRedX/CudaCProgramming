#include <cstdlib>
#include <string>
#include <ctime>
#include <iostream>

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int)time(&t));
    for(int i=0;i<size;++i){
        ip[i] = (float)( rand() & 0xFF ) / 10.0F;
    }
}

void sumArrayOnHost(float *A, float *B, float *C, const int N){
    for(int idx = 0;idx < N; ++idx){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){

    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    auto h_A = (float *) malloc(nBytes);
    auto h_B = (float *) malloc(nBytes);
    auto h_C = (float *) malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArrayOnHost(h_A, h_B, h_C, nElem);

    for(int i=0;i<nElem;++i){
        std::cout << "i: " << h_C[i] << "\n";
    }

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}