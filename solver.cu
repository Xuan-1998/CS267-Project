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

void add_source(int N, float *x, float *s, float dt)
{
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++)
        x[i] += dt * s[i];
}

void set_bnd(int N, int b, float *x)
{
    int i;
    for (i = 1; i <= N; i++)
    {
        x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }
    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}


__global__ void projectHelper1(int N, float *u, float *v, float *p, float *div) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid % N + 1, j = tid / N + 1;
    
    // printf();
    float h;
    h = 1.0 / N;
    
    if (i <= N && j <= N) {
        // printf("rows is: %d, cols is %d\n", i, j);
        div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
        p[IX(i, j)] = 0;
    } else {
        printf("%d\n", tid);
    }
    __syncthreads();
}
void project(int N, float *u, float *v, float *p, float *div, float *p_new)
{
    int i, j, k;
    float h;
    h = 1.0 / N;

    const int size = (N + 2) * (N + 2);
    int mem_size = size*sizeof(float);

    float *d_u = nullptr;
    float *d_v = nullptr;
    float *d_p = nullptr;
    float *d_div = nullptr;

    cudaMalloc(&d_u, sizeof(u));
    cudaMemcpy(d_u, u, sizeof(u), cudaMemcpyHostToDevice);
    cudaMalloc(&d_v, sizeof(v));
    cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice);
    cudaMalloc(&d_p, sizeof(p));
    cudaMemcpy(d_p, p, sizeof(p), cudaMemcpyHostToDevice);
    cudaMalloc(&d_div, sizeof(div));
    cudaMemcpy(d_div, div, sizeof(div), cudaMemcpyHostToDevice);
    // dim3 gridDim(N / 16, N / 16);
    // dim3 blockDim(16, 16);
    std::cout << "It comes before projectHelper1<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, u, v, p, div);\n";
    projectHelper1<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, u, v, p, div);
    std::cout << "It comes after projectHelper1<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(N, u, v, p, div);\n";
    // cudaDeviceSynchronize();
    // std::cout << "It comes after cudaDeviceSynchronize";
    cudaMemcpy(u, d_u, sizeof(u), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, sizeof(v), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_p, sizeof(d_p), cudaMemcpyDeviceToHost);
    cudaMemcpy(div, d_div, sizeof(div), cudaMemcpyDeviceToHost);
    std::cout << "It comes all the way down of copy\n";
    // cudaFree(d_u);
    // cudaFree(d_v);
    // cudaFree(d_p);
    // cudaFree(d_div);
    // let's don't free
    std::cout << "It comes all the way down of free\n";

    // for (i = 1; i <= N; i++)
    // {
    //     for (j = 1; j <= N; j++)
    //     {
    //         div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
    //         p[IX(i, j)] = 0;
    //     }
    // }
    
    

    set_bnd(N, 0, div);
    set_bnd(N, 0, p);
    // std::cout << "It comes all the way down of free";

    std::cout << "It goes after bnd setting";

    for (k = 0; k < 20; k++)
    {

        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                p_new[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                                   p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
                                  4;
            }
        }
        SWAP(p, p_new);
        set_bnd(N, 0, p);
    }
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
        {
            u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
        }
    }
    set_bnd(N, 1, u);
    set_bnd(N, 2, v);
}

void diffuse(int N, int b, float *x, float *x0, float diff, float dt)
{
    int i, j, k;
    float a = dt * diff * N * N;
    // std::cout << "a:" << a << ", dt:" << dt << ", diff:" << diff << std::endl;
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
        {
            x[IX(i, j)] = x0[IX(i, j)] + a * (x0[IX(i + 1, j)] + x0[IX(i - 1, j)] + x0[IX(i, j + 1)] + x0[IX(i, j - 1)] - 4 * x0[IX(i, j)]);
        }
    }
    set_bnd(N, b, x);
}

void advect(int N, int b, float *d, float *d0, float *u, float *v, float dt)
{
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;
    dt0 = dt * N;
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
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
    }
    set_bnd(N, b, d);
}

void dens_step(int N, float *x, float *x0, float *u, float *v, float diff,
               float dt)
{
    add_source(N, x, x0, dt);
    SWAP(x0, x);
    diffuse(N, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(N, 0, x, x0, u, v, dt);
}
// num of threads = 256
void vel_step(int N, float *u, float *v, float *u0, float *v0,
              float visc, float dt, float *p_new)
{
    add_source(N, u, u0, dt);
    add_source(N, v, v0, dt);
    SWAP(u0, u);
    diffuse(N, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(N, 2, v, v0, visc, dt);
    // dim3 gridDim(N/16, N/16)
    // dim3 blockDim(16, 16)
    // row = blockIdx.y * blockDim.y + threadIdx.y;
    // col = blockIdx.x * blockDim.x + threadIdx.x;
    // u[row][col]
    project(N, u, v, u0, v0, p_new);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(N, 1, u, u0, u0, v0, dt);
    advect(N, 2, v, v0, u0, v0, dt);
    project(N, u, v, u0, v0, p_new);
}

int main()
{
    auto start_time = std::chrono::steady_clock::now();
    int simulating = 1;
    const int N = 100;
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
    while (simulating--)
    {
        // get_from_UI(dens_prev, u_prev, v_prev);
        vel_step(N, u, v, u_prev, v_prev, visc, dt, p_new);
        dens_step(N, dens, dens_prev, u, v, diff, dt);
        using namespace std;
        //cout << u[5] << endl;
        //  draw_dens(N, dens);
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> difference = end_time - start_time;
    double seconds = difference.count();
    std::cout << "Simulation Time = " << seconds << " seconds for " << N << " blocks.\n";
}