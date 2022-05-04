#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <stdio.h>

#define IX(i, j) ((i) + (N + 2) * (j))
#define SWAP(x0, x)      \
    {                    \
        float *tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }

#define NUM_THREADS 256

// __device__ void SWAP(float *x0, float *x) 
// {
//         float *tmp = x0; \
//         x0 = x;          \
//         x = tmp;         \
// }

__global__ void inline add_source(int N, float *x, float *s, float dt)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int i, size = (N + 2) * (N + 2);
    if (tid < (N + 2) * (N + 2)) 
    {
        x[tid] += dt * s[tid];
    }
    // for (i = 0; i < size; i++)
    //     x[i] += dt * s[i];
    __syncthreads();
}

// __global__ void set_bnd_helper(int N, int b, float *x) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
//     if (i <= N) {
//         x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
//         x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
//         x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
//         x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
//         __syncthreads();
//     }
// }


__global__ void inline set_bnd(int N, int b, float *x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    // int i;

    // set_bnd_helper<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, b, x);
    
    // int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    // if (i <= N) {
    //     x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
    //     x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
    //     x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    //     x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    //     __syncthreads();
    // }
    if (i <= N)
    {
        x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];       
    }

    // for (i = 1; i <= N; i++)
    // {
    //     x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
    //     x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
    //     x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    //     x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    // }

    // x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    // x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
    // x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    // x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
    __syncthreads();
}

__global__ void inline set_bnd_finish(int N, int b, float*x) 
{
    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
    __syncthreads();
}


__global__ void inline projectHelper1(int N, float *u, float *v, float *p, float *div) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid % N + 1, j = tid / N + 1;
    
    // printf();
    float h;
    h = 1.0 / N;
    
    if (i <= N && j <= N) {
        // printf("rows is: %d, cols is %d\n", i, j);
        div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
        p[IX(i, j)] = 0;
    }
    // } else {
    //     printf("%d\n", tid);
    // }
    __syncthreads();
}

__global__ void inline projectHelper2(int N, float *div, float *p, float *p_new) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid % N + 1, j = tid / N + 1;

    if (i <= N && j <= N) {
        p_new[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                    p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
                    4;
    }
    __syncthreads();

}

__global__ void inline projectHelper3(int N, float *u, float *v, float *p) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid % N + 1, j = tid / N + 1;
    float h;
    h = 1.0 / N;
    if (i <= N && j <= N) {

        u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
        v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
    }
    __syncthreads();

}


void inline project(int N, float *u, float *v, float *p, float *div, float *p_new)
{
    // int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float h;
    int k;
    h = 1.0 / N;

    // int i = tid % N + 1, j = tid / N + 1;
    // if (i <= N && j <= N) {
    //     // printf("rows is: %d, cols is %d\n", i, j);
    //     div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
    //     p[IX(i, j)] = 0;
    // }
    // __syncthreads();
    const int size = (N + 2) * (N + 2);
    
    projectHelper1<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, u, v, p, div);

    // for (i = 1; i <= N; i++)
    // {
    //     for (j = 1; j <= N; j++)
    //     {
    //         div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
    //         p[IX(i, j)] = 0;
    //     }
    // }
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 0, div);
    set_bnd_finish<<<1, 1>>>(N, 0, div);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 0, p);
    set_bnd_finish<<<1, 1>>>(N, 0, p);
    for (k = 0; k < 20; k++)
    {

        // if (i <= N && j <= N) {
        //     p_new[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
        //                 p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
        //                 4;
        // }

        // __syncthreads();
        projectHelper2<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, div, p, p_new);
        SWAP(p, p_new);
        set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 0, p);
        set_bnd_finish<<<1, 1>>>(N, 0, p);
    }

    // if (i <= N && j <= N) {
    //     u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
    //     v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
    // }
    projectHelper3<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, u, v, p);

    // __syncthreads();
    // for (i = 1; i <= N; i++)
    // {
    //     for (j = 1; j <= N; j++)
    //     {
    //         u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
    //         v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
    //     }
    // }
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 1, u);
    set_bnd_finish<<<1, 1>>>(N, 1, u);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 2, v);
    set_bnd_finish<<<1, 1>>>(N, 2, v);
}

// void set_bnd(int N, int b, float *x)
// {
//     // int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int i;

//     // set_bnd_helper<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, b, x);
    
//     // int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
//     // if (i <= N) {
//     //     x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
//     //     x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
//     //     x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
//     //     x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
//     //     __syncthreads();
//     // }
//     for (i = 1; i <= N; i++)
//     {
//         x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
//         x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
//         x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
//         x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
//     }
//     x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
//     x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
//     x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
//     x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
// }


__global__ void inline diffuse(int N, int b, float *x, float *x0, float diff, float dt)
{
    // int i, j, k;
    float a = dt * diff * N * N;
    // std::cout << "a:" << a << ", dt:" << dt << ", diff:" << diff << std::endl;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid % N + 1;
    int j = tid / N + 1;
    if (i <= N && j <= N) 
    {
      x[IX(i, j)] = x0[IX(i, j)] + a * (x0[IX(i + 1, j)] + x0[IX(i - 1, j)] + x0[IX(i, j + 1)] + x0[IX(i, j - 1)] - 4 * x0[IX(i, j)]);  
    }
    // for (i = 1; i <= N; i++)
    // {
    //     for (j = 1; j <= N; j++)
    //     {
    //         x[IX(i, j)] = x0[IX(i, j)] + a * (x0[IX(i + 1, j)] + x0[IX(i - 1, j)] + x0[IX(i, j + 1)] + x0[IX(i, j - 1)] - 4 * x0[IX(i, j)]);
    //     }
    // }
    // set_bnd(N, b, x);
    __syncthreads();
}
// remember: for all the diffuse, after that you need to set_bnd

__global__ void inline advect(int N, int b, float *d, float *d0, float *u, float *v, float dt)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid % N + 1;
    int j = tid / N + 1;
    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;
    dt0 = dt * N;
    if (i <= N && j <= N) 
    {
        x = i - dt0 * u[IX(i, j)];
        y = j - dt0 * v[IX(i, j)];
        if (x < 0.5)
            x = 0.5;
        if (x > N + 0.5)
            x = N + 0.5;
        i0 = (int)x;
        i1 = i0 + 1;
        if (y < 0.5)
            y = 0.5;
        if (y > N + 0.5)
            y = N + 0.5;
        j0 = (int)y;
        j1 = j0 + 1;
        s1 = x - i0;
        s0 = 1 - s1;
        t1 = y - j0;
        t0 = 1 - t1;
        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                        s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);        
    }
    // for (i = 1; i <= N; i++)
    // {
    //     for (j = 1; j <= N; j++)
    //     {
    //         x = i - dt0 * u[IX(i, j)];
    //         y = j - dt0 * v[IX(i, j)];
    //         if (x < 0.5)
    //             x = 0.5;
    //         if (x > N + 0.5)
    //             x = N + 0.5;
    //         i0 = (int)x;
    //         i1 = i0 + 1;
    //         if (y < 0.5)
    //             y = 0.5;
    //         if (y > N + 0.5)
    //             y = N + 0.5;
    //         j0 = (int)y;
    //         j1 = j0 + 1;
    //         s1 = x - i0;
    //         s0 = 1 - s1;
    //         t1 = y - j0;
    //         t0 = 1 - t1;
    //         d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
    //                       s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    //     }
    // }
    // set_bnd(N, b, d);
    __syncthreads();
}

// remember: for all the advect, after that you need to set_bnd


void inline dens_step(int N, float *x, float *x0, float *u, float *v, float diff,
               float dt)
{
    int size = (N + 2) * (N + 2);
    add_source<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, x, x0, dt);
    SWAP(x0, x);
    diffuse<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 0, x, x0, diff, dt);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 0, x);
    set_bnd_finish<<<1, 1>>>(N, 0, x);
    SWAP(x0, x);
    advect<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 0, x, x0, u, v, dt);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 0, x);
    set_bnd_finish<<<1, 1>>>(N, 0, x);
}

void inline vel_step(int N, float *u, float *v, float *u0, float *v0,
              float visc, float dt, float *p_new)
{
    const int size = (N + 2) * (N + 2);
    add_source<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, u, u0, dt);
    add_source<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, v, v0, dt);
    SWAP(u0, u);
    diffuse<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 1, u, u0, visc, dt);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 1, u);
    set_bnd_finish<<<1, 1>>>(N, 1, u);
    SWAP(v0, v);
    diffuse<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 2, v, v0, visc, dt);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 2, v);
    set_bnd_finish<<<1, 1>>>(N, 2, v);   
    project(N, u, v, u0, v0, p_new);
    // project(N, u, v, u0, v0, p_new);
    SWAP(u0, u);
    SWAP(v0, v);
    advect<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 1, u, u0, u0, v0, dt);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 1, u);
    set_bnd_finish<<<1, 1>>>(N, 1, u);   
    advect<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 2, v, v0, u0, v0, dt);
    set_bnd<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, 2, v);
    set_bnd_finish<<<1, 1>>>(N, 2, v);  
    project(N, u, v, u0, v0, p_new);
    // project(N, u, v, u0, v0, p_new);
}

int main()
{
    auto start_time = std::chrono::steady_clock::now();
    int simulating = 100;
    const int N = 400;
    const int size = (N + 2) * (N + 2);
    float static u[size], v[size];
    float static u_prev[size]; // = {[0 ... 15] = 1000.0};
    float static v_prev[size]; // = {[0 ... 15] = 1000.0};
    float static dens[size], dens_prev[size];
    float static p_new[size];

    std::fill(u_prev, u_prev + size, 100.0);
    std::fill(v_prev, v_prev + size, 100.0);
    std::fill(dens_prev, dens_prev + size, 100.0);
    float dt = 0.01;
    float visc = 0.00001;
    float diff = 0.00001;

    float *d_u = nullptr;
    float *d_v = nullptr;
    float *d_u_prev = nullptr;
    float *d_v_prev = nullptr;
    float *d_dens = nullptr;
    float *d_dens_prev = nullptr;
    float *d_p_new = nullptr;


    cudaMalloc(&d_u, sizeof(u));
    cudaMemcpy(d_u, u, sizeof(u), cudaMemcpyHostToDevice);
    cudaMalloc(&d_v, sizeof(v));
    cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice);
    cudaMalloc(&d_u_prev, sizeof(u_prev));
    cudaMemcpy(d_u_prev, u_prev, sizeof(u_prev), cudaMemcpyHostToDevice);
    cudaMalloc(&d_v_prev, sizeof(v_prev));
    cudaMemcpy(d_v_prev, v_prev, sizeof(v_prev), cudaMemcpyHostToDevice);
    cudaMalloc(&d_dens, sizeof(dens));
    cudaMemcpy(d_dens, dens, sizeof(dens), cudaMemcpyHostToDevice);
    cudaMalloc(&d_dens_prev, sizeof(dens_prev));
    cudaMemcpy(d_dens_prev, dens_prev, sizeof(dens_prev), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p_new, sizeof(p_new));
    cudaMemcpy(d_p_new, p_new, sizeof(p_new), cudaMemcpyHostToDevice);



    while (simulating--)
    {
        // get_from_UI(dens_prev, u_prev, v_prev);
        vel_step(N, d_u, d_v, d_u_prev, d_v_prev, visc, dt, d_p_new);
        dens_step(N, d_dens, d_dens_prev, d_u, d_v, diff, dt);
        using namespace std;
        cudaDeviceSynchronize();
        //cout << u[5] << endl;
        //  draw_dens(N, dens);
    }
    // cudaDeviceSynchronize();

    cudaMemcpy(u, d_u, sizeof(u), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, sizeof(v), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_prev, d_u_prev, sizeof(u_prev), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_prev, d_v_prev, sizeof(v_prev), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens, d_dens, sizeof(dens), cudaMemcpyDeviceToHost);
    cudaMemcpy(dens_prev, d_dens_prev, sizeof(dens_prev), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_new, d_p_new, sizeof(p_new), cudaMemcpyDeviceToHost);






    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> difference = end_time - start_time;
    double seconds = difference.count();
    std::cout << "Simulation Time = " << seconds << " seconds for " << N << " blocks.\n";
}