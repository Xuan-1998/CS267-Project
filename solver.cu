#include <cuda.h>

#include <iostream>
#include <algorithm>

using floating_t = double;
#define IX(i, j) ((i) + (N + 2) * (j))
#define SWAP(x0, x)           \
    {                         \
        floating_t *tmp = x0; \
        x0 = x;               \
        x = tmp;              \
    }

#define NUM_THREADS 256
static int N_blks_small, N_blks_large, N_blks_bnd;

__global__ void add_source(int N, floating_t *x, floating_t *s, floating_t dt)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > (N + 2) * (N + 2))
        return;
    x[tid] += dt * s[tid];
    /*
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++)
        x[i] += dt * s[i];
        */
}

__global__ void set_bnd_helper(int N, int b, floating_t *x)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > N)
        return;
    const int i = tid + 1; // 1 <= i <= N

    x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
    x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
}
void set_bnd(int N, int b, floating_t *x)
{
    // TODO: templatize
    set_bnd_helper<<<N_blks_bnd, NUM_THREADS>>>(N, b, x);

    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

__global__ void diffuse_helper(int N, int b, floating_t *x, floating_t *x0, floating_t diff, floating_t dt, floating_t a)
{
    // TODO: remove computation of i and j and just have tid control everything
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > N * N)
        return;
    const int i = tid % N + 1, j = tid / N + 1;
    x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                  (1 + 4 * a);
}

void diffuse(int N, int b, floating_t *x, floating_t *x0, floating_t diff, floating_t dt)
{
    floating_t a = dt * diff * N * N;
    for (int k = 0; k < 20; k++)
    {
        diffuse_helper<<<N_blks_small, NUM_THREADS>>>(N, b, x, x0, diff, dt, a);
        set_bnd(N, b, x);
    }
}

void advect(int N, int b, floating_t *d, floating_t const *d0, floating_t const *u, floating_t const *v, floating_t dt)
{
    int i, j, i0, j0, i1, j1;
    floating_t x, y, s0, t0, s1, t1;
    const floating_t dt0 = dt * N;
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

__global__ void project_helper1(int N, floating_t *u, floating_t *v, floating_t *p, floating_t *div, floating_t h)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > N * N)
        return;
    const int i = tid % N + 1, j = tid / N + 1;
    div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
    p[IX(i, j)] = 0;
}
__global__ void project_helper2(int N, floating_t *u, floating_t *v, floating_t *p, floating_t *div)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > N * N)
        return;
    const int i = tid % N + 1, j = tid / N + 1;
    p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                   p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
                  4;
}
__global__ void project_helper3(int N, floating_t *u, floating_t *v, floating_t *p, floating_t *div, floating_t h)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > N * N)
        return;
    const int i = tid % N + 1, j = tid / N + 1;
    u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
    v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
}
void project(int N, floating_t *u, floating_t *v, floating_t *p, floating_t *div)
{
    floating_t h = 1.0 / N;
    /*
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
        {
            div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
            p[IX(i, j)] = 0;
        }
    }
    */
    project_helper1<<<N_blks_small, NUM_THREADS>>>(N, u, v, p, div, h);
    set_bnd(N, 0, div);
    set_bnd(N, 0, p);
    for (int k = 0; k < 20; k++)
    {
        /*
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                               p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
                              4;
            }
        }
        */
        project_helper2<<<N_blks_small, NUM_THREADS>>>(N, u, v, p, div);
        set_bnd(N, 0, p);
    }
    /*
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
        {
            u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
        }
    }
    */
    project_helper3<<<N_blks_small, NUM_THREADS>>>(N, u, v, p, div, h);
    set_bnd(N, 1, u);
    set_bnd(N, 2, v);
}

void vel_step(int N, floating_t *u, floating_t *v, floating_t *u0, floating_t *v0,
              floating_t visc, floating_t dt)
{
    add_source<<<N_blks_large, NUM_THREADS>>>(N, u, u0, dt);
    add_source<<<N_blks_large, NUM_THREADS>>>(N, v, v0, dt);
    SWAP(u0, u);
    diffuse(N, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(N, 2, v, v0, visc, dt);
    project(N, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(N, 1, u, u0, u0, v0, dt);
    advect(N, 2, v, v0, u0, v0, dt);
    project(N, u, v, u0, v0);
}

void dens_step(int N, floating_t *x, floating_t *x0, floating_t *u, floating_t *v, floating_t diff,
               floating_t dt)
{
    add_source<<<N_blks_large, NUM_THREADS>>>(N, x, x0, dt);
    SWAP(x0, x);
    diffuse(N, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(N, 0, x, x0, u, v, dt);
}

int main()
{
    int simulating = 1000;
    const int N = 1000;
    const int size = (N + 2) * (N + 2);
    N_blks_bnd = (N + NUM_THREADS - 1) / NUM_THREADS;
    N_blks_small = (N * N + NUM_THREADS - 1) / NUM_THREADS;
    N_blks_large = (size + NUM_THREADS - 1) / NUM_THREADS;
    /*
    floating_t static u[size]{}, v[size]{};
    floating_t static u_prev[size]{}; // = {[0 ... 15] = 1000.0};
    floating_t static v_prev[size]{}; // = {[0 ... 15] = 1000.0};
    floating_t static dens[size]{}, dens_prev[size]{};

    std::fill(u_prev, u_prev + size, 100.0);
    std::fill(v_prev, v_prev + size, 100.0);
    std::fill(dens_prev, dens_prev + size, 100.0);
    */
    floating_t *u, *v, *u_prev, *v_prev, *dens, *dens_prev;
    cudaMalloc((void **)&u, size * sizeof(floating_t));
    cudaMalloc((void **)&v, size * sizeof(floating_t));
    cudaMalloc((void **)&u_prev, size * sizeof(floating_t));
    cudaMalloc((void **)&v_prev, size * sizeof(floating_t));
    cudaMalloc((void **)&dens, size * sizeof(floating_t));
    cudaMalloc((void **)&dens_prev, size * sizeof(floating_t));

    /* fill commands here */

    constexpr floating_t dt = 0.01;
    constexpr floating_t visc = 0.1;
    constexpr floating_t diff = 1;
    while (simulating--)
    {
        // get_from_UI(dens_prev, u_prev, v_prev);
        vel_step(N, u, v, u_prev, v_prev, visc, dt);
        dens_step(N, dens, dens_prev, u, v, diff, dt);
        using namespace std;
        cout << u[5] << endl;
        // draw_dens(N, dens);
    }
}