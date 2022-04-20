#include <cuda.h> 

#include <iostream>
#include <algorithm>

using floating_t = double;
#define IX(i, j) ((i) + (N + 2) * (j))
#define SWAP(x0, x)      \
    {                    \
        floating_t *tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }


__device__ floating_t testvar = 5;

/*
const int size = N;
*/
void project(int N, floating_t *u, floating_t *v, floating_t *p, floating_t *div);

void add_source(int N, floating_t *x, floating_t *s, floating_t dt)
{
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++)
        x[i] += dt * s[i];
}

void set_bnd(int N, int b, floating_t *x)
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

void diffuse(int N, int b, floating_t *x, floating_t *x0, floating_t diff, floating_t dt)
{
    int i, j, k;
    floating_t a = dt * diff * N * N;
    for (k = 0; k < 20; k++)
    {
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                   x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                              (1 + 4 * a);
            }
        }
        set_bnd(N, b, x);
    }
}

void advect(int N, int b, floating_t *d, floating_t *d0, floating_t *u, floating_t *v, floating_t dt)
{
    int i, j, i0, j0, i1, j1;
    floating_t x, y, s0, t0, s1, t1, dt0;
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

void dens_step(int N, floating_t *x, floating_t *x0, floating_t *u, floating_t *v, floating_t diff,
               floating_t dt)
{
    add_source(N, x, x0, dt);
    SWAP(x0, x);
    diffuse(N, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(N, 0, x, x0, u, v, dt);
}

void vel_step(int N, floating_t *u, floating_t *v, floating_t *u0, floating_t *v0,
              floating_t visc, floating_t dt)
{
    add_source(N, u, u0, dt);
    add_source(N, v, v0, dt);
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

void project(int N, floating_t *u, floating_t *v, floating_t *p, floating_t *div)
{
    int i, j, k;
    floating_t h;
    h = 1.0 / N;
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
        {
            div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(N, 0, div);
    set_bnd(N, 0, p);
    for (k = 0; k < 20; k++)
    {
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                               p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
                              4;
            }
        }
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

int main()
{
    int simulating = 1000;
    const int N = 1000;
    const int size = (N + 2) * (N + 2);
    floating_t static u[size]{}, v[size]{};
    floating_t static u_prev[size]{}; // = {[0 ... 15] = 1000.0};
    floating_t static v_prev[size]{}; // = {[0 ... 15] = 1000.0};
    floating_t static dens[size]{}, dens_prev[size]{};

    std::fill(u_prev, u_prev + size, 100.0);
    std::fill(v_prev, v_prev + size, 100.0);
    std::fill(dens_prev, dens_prev + size, 100.0);
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