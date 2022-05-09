#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <mpi.h>

#define IX(i, j) ((i) + (N + 2) * (j))
#define SWAP(x0, x)      \
    {                    \
        float *tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }

using std::vector;

const int N = 2;
const int size = (N + 2) * (N + 2);
float static u[size]{}, v[size]{};
float static u_prev[size]{}; // = {[0 ... 15] = 1000.0};
float static v_prev[size]{}; // = {[0 ... 15] = 1000.0};
float static dens[size]{}, dens_prev[size]{};

MPI_Comm cart;
int rankN;
int myRank;
int procDim;
int numRows, numCols;
int procCoords[2]{}, dims[2]{};
int left, right, top, bot;
MPI_Datatype col_t, row_t;
enum
{
    LEFT,
    RIGHT,
    DOWN,
    UP,
    CENTER
};
int neighborProcs[4]{MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

void communicateBorders(float *ptr);
void project(int N, float *u, float *v, float *p, float *div);
void foreach (int iterations, std::function<void(int, int)> subroutine)
{
    int leftBound = left, rightBound = right, topBound = top, botBound = bot;
    if (procCoords[0] == 0)
    {
        leftBound++;
    }
    else if (procCoords[0] == procDim - 1)
    {
        rightBound--;
    }
    if (procCoords[1] == 0)
    {
        botBound++;
    }
    else if (procCoords[1] == i - 1)
    {
        topBound--;
    }
    for (int k = 0; k < iteratkons; k++)
    {
        for (int i = botBound; i < topBound; i++)
        {
            for (int j = leftBound; j < rightBound; j++)
            {
                subroutine(i, j);
            }
        }
    }
}

void add_source(int N, float *x, float *s, float dt)
{
    // perfect to gpu-ize?
    // int i, size = (N + 2) * (N + 2);
    // for (i = 0; i < size; i++)
    // x[i] += dt * s[i];
    foreach (1, [=](int i, int j)
             { x[IX(i, j)] += dt * s[IX(i, j)]; })
        ;
}

void set_bnd(int N, int b, float *x)
{
    // TODO: if not border, then return, i.e. parallelize
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

void diffuse(int N, int b, float *x, float *x0, float diff, float dt)
{
    // TODO: red-black re-ordering
    int i, j, k;
    float a = dt * diff * N * N;
    for (k = 0; k < 20; k++)
    {
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                // TODO: MAKE THIS JACOBI
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                   x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                              (1 + 4 * a);
            }
        }
        set_bnd(N, b, x);
    }
}

void advect(int N, int b, float *d, float *d0, float *u, float *v, float dt)
{
    // u, v read only
    /*
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
    */
    foreach (1, [=](int i, int j)
             {
                int i0, j0, i1, j1;
                float x, y, s0, t0, s1, t1, dt0;
                dt0 = dt * N;

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
                            s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]); })
        ;
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

void vel_step(int N, float *u, float *v, float *u0, float *v0,
              float visc, float dt)
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

void project(int N, float *u, float *v, float *p, float *div)
{
    // red-black
    int i, j, k;
    const float h = 1.0 / N;
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
    foreach (1, [=](int i, int j)
             {
                div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
                p[IX(i, j)] = 0; })
        ;
    set_bnd(N, 0, div);
    set_bnd(N, 0, p);
    // TODO: Jacobi-ze this
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
    foreach (1, [=](int i, int j)
             {
            u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h; })
        ;
    set_bnd(N, 1, u);
    set_bnd(N, 2, v);
}

void communicateBorders(float *ptr)
{
    vector<MPI_Request> requests{};
    if (neighborProcs[LEFT] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Isend(ptr, 1, col_t, neighborProcs[LEFT], 0, cart, &requests.back());
    }
    if (neighborProcs[RIGHT] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Isend(ptr + (numCols - 1), 1, col_t, neighborProcs[RIGHT], 1, cart, &requests.back());
    }
    if (neighborProcs[UP] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Isend(ptr, 1, col_t, neighborProcs[UP], 2, cart, &requests.back());
    }
    if (neighborProcs[DOWN] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Isend(ptr + (numCols - 1) * numRows, 1, col_t, neighborProcs[DOWN], 3, cart, &requests.back());
    }
    vector<MPI_Status> statuses{};
    if (neighborProcs[LEFT] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Irecv(ptr, 1, col_t, neighborProcs[LEFT], 1, cart, &requests.back());
    }
    if (neighborProcs[RIGHT] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Irecv(ptr + (numCols - 1), 1, col_t, neighborProcs[RIGHT], 0, cart, &requests.back());
    }
    if (neighborProcs[UP] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Irecv(ptr, 1, col_t, neighborProcs[UP], 3, cart, &requests.back());
    }
    if (neighborProcs[DOWN] != MPI_PROC_NULL)
    {
        requests.emplace_back();
        MPI_Irecv(ptr + (numCols - 1) * numRows, 1, col_t, neighborProcs[DOWN], 2, cart, &requests.back());
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void factorize(const int n)
{
    // MPI cart
    MPI_Comm_size(MPI_COMM_WORLD, &rankN);
    // assume perfect powers of 4 for both processors, powers of 2 for (n + 2)
    procDim = sqrt(rankN);
    dims[0] = procDim, dims[1] = procDim;
    const int periods[2]{false, false};
    MPI_Cart_create(MPI_COMM_WORLD, 2, periods, true, &cart);
    MPI_Comm_rank(cart, &myRank);
    MPI_Cart_coords(cart, myRank, 2, procCoords);
    numRows = numCols = (n + 2) / procDim;
}

void assignBlocks(const int n)
{
    left = procCoords[0] * numRows;
    right = (procCoords[0] + 1) * numRows;
    bot = procCoords[1] * numRows;
    top = (procCoords[1] + 1) * numRows;
    // j = column, i = row
    MPI_Type_contiguous(numCols, MPI_FLOAT, &row_t);
    MPI_Type_vector(1, numRows, numCols /*stride */, MPI_FLOAT, &col_t);
    MPI_Type_commit(row_t);
    MPI_Type_commit(col_t);
    MPI_Cart_shift(cart, 0, 1, &neighborProcs[LEFT], &neighborProcs[RIGHT]);
    MPI_Cart_shift(cart, 1, 1, &neighborProcs[DOWN], &neighborProcs[UP]);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int simulating = 100;

    std::fill(u_prev, u_prev + size, 100.0);
    std::fill(v_prev, v_prev + size, 100.0);
    std::fill(dens_prev, dens_prev + size, 100.0);
    float dt = 0.01;
    float visc = 0.1;
    float diff = 1;
    // MPI broadcast
    while (simulating--)
    {
        // get_from_UI(dens_prev, u_prev, v_prev);
        vel_step(N, u, v, u_prev, v_prev, visc, dt);
        dens_step(N, dens, dens_prev, u, v, diff, dt);
        using namespace std;
        cout << u[5] << endl;
        // draw_dens(N, dens);
    }
    MPI_Finalize();
}