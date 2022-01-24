#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <getopt.h>
#include <time.h>
#include <mpi.h>

#include "solve.h"


#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define RIGHT_BORBER 4.0
#define TOP_BORDER 3.0


int set_best_config(struct RunConfig *run_config)
{
    int error_code =  MPI_ERR_UNKNOWN;
    if (run_config->mpi_config->num_procs != 1) {
        double best_k = 3;
        // dim_x - количество процессов на направлении x
        for (int dim_x = 1; dim_x <= run_config->mpi_config->num_procs; ++dim_x) {
            if (run_config->mpi_config->num_procs % dim_x != 0) {
                continue;
            }
            int dim_y = run_config->mpi_config->num_procs / dim_x;
            double k = (double) run_config->m / dim_x / run_config->n * dim_y;
            //printf("k == %.4lf %d %d\n", k, run_config->m / dim_x, run_config->n / dim_y);
            if (k >= 0.5 && k <= 2 && k < best_k) {
                best_k = k;
                run_config->mpi_config->dims[0] = dim_x;
                run_config->mpi_config->dims[1] = dim_y;
            }
        }
        //printf("procs count: %dx%d, domain size: %dx%d\n", run_config->mpi_config->dims[0], run_config->mpi_config->dims[1], run_config->domain_m, run_config->domain_n);
    } else {
        run_config->mpi_config->dims[0] = 1;
        run_config->mpi_config->dims[1] = 1;
    }
    if (run_config->mpi_config->dims[0] == 0 || run_config->mpi_config->dims[1] == 0) {
        return MPI_ERR_UNKNOWN;
    }
    
    int periods[2] = { 0 };
    error_code = MPI_Cart_create(MPI_COMM_WORLD, 2, run_config->mpi_config->dims, periods, 1, &run_config->mpi_config->comm);
    CHECK_ERROR_CODE(error_code);
    
    error_code = MPI_Comm_rank(run_config->mpi_config->comm, &run_config->mpi_config->proc_id);
    CHECK_ERROR_CODE(error_code);
    
    error_code = MPI_Cart_coords(run_config->mpi_config->comm, run_config->mpi_config->proc_id, 2, run_config->mpi_config->proc_coords);
    CHECK_ERROR_CODE(error_code);
    
    error_code = MPI_Cart_shift(run_config->mpi_config->comm, 0, 1, &run_config->mpi_config->left_x_id, &run_config->mpi_config->right_x_id);
    CHECK_ERROR_CODE(error_code);

    error_code = MPI_Cart_shift(run_config->mpi_config->comm, 1, 1, &run_config->mpi_config->bottom_y_id, &run_config->mpi_config->top_y_id);
    CHECK_ERROR_CODE(error_code);
    
    // Базовый размер домена, можно использовать для расчёта start
    run_config->domain_m = run_config->m / run_config->mpi_config->dims[0];
    run_config->domain_n = run_config->n / run_config->mpi_config->dims[1];
    my_int remainder_x = run_config->m % run_config->mpi_config->dims[0];
    my_int remainder_y = run_config->n % run_config->mpi_config->dims[1];
    run_config->start_i = run_config->domain_m * run_config->mpi_config->proc_coords[0] + MIN(remainder_x, run_config->mpi_config->proc_coords[0]);
    run_config->start_j = run_config->domain_n * run_config->mpi_config->proc_coords[1] + MIN(remainder_y, run_config->mpi_config->proc_coords[1]);
    // Реальный размер домена, можно использовать для расчёта stop
    if (run_config->mpi_config->proc_coords[0] < remainder_x) {
        ++run_config->domain_m;
    }
    if (run_config->mpi_config->proc_coords[1] < remainder_y) {
        ++run_config->domain_n;
    }
    run_config->stop_i = run_config->start_i + run_config->domain_m - 1;
    run_config->stop_j = run_config->start_j + run_config->domain_n - 1;
    
    run_config->h1 = RIGHT_BORBER / (run_config->m - 1);
    run_config->h2 = TOP_BORDER / (run_config->n - 1);
    run_config->inv_h1 = 1 / run_config->h1;
    run_config->inv_h2 = 1 / run_config->h2;
    run_config->sqinv_h1 = run_config->inv_h1 * run_config->inv_h1;
    run_config->sqinv_h2 = run_config->inv_h2 * run_config->inv_h2;
    
    if (run_config->mpi_config->proc_id == 0) {
        printf("procs count: %dx%d, domain size: %ldx%ld\n", run_config->mpi_config->dims[0], run_config->mpi_config->dims[1], run_config->domain_m, run_config->domain_n);
    }
    
    error_code = MPI_SUCCESS;
done:
    return error_code;
}

int main(int argc, char **argv)
{
    int error_code =  MPI_ERR_UNKNOWN;
    
    error_code = MPI_Init(&argc, &argv);
    CHECK_ERROR_CODE(error_code);
    
    struct MpiConfig mpi_config = { 0 };
    struct RunConfig run_config = { 0 };
    run_config.mpi_config = &mpi_config;
    
    error_code = MPI_Comm_size(MPI_COMM_WORLD, &run_config.mpi_config->num_procs);
    CHECK_ERROR_CODE(error_code);
    
    error_code = MPI_Comm_rank(MPI_COMM_WORLD, &run_config.mpi_config->proc_id);
    CHECK_ERROR_CODE(error_code);
    
    int opt;
    run_config.eps = 1e-6;
    
    while ((opt = getopt(argc, argv, "m:n:e:p:g")) != -1) {
        switch (opt) {
        case 'm':
            run_config.m = atoi(optarg);
            break;
        case 'n':
            run_config.n = atoi(optarg);
            break;
        case 'e':
            run_config.eps = atof(optarg);
            break;
        case 'p':
            printf("########## WARNING! THIS IS DEBUG OPTION! ##########\n");
            run_config.mpi_config->num_procs = atoi(optarg);
            break;
        default:
            if (run_config.mpi_config->proc_id == 0) {
                printf("Unknown arg: %c\n", opt);
            }
            error_code = MPI_ERR_UNKNOWN;
            goto done;
        }
    }
    
    error_code = set_best_config(&run_config);
    if (error_code) {
        if (run_config.mpi_config->proc_id == 0) {
            printf("set_best_config()\n");
        }
        error_code = MPI_ERR_UNKNOWN;
        goto done;
    }
    
    error_code = alloc_data(&run_config);
    if (error_code) {
        if (run_config.mpi_config->proc_id == 0) {
            printf("alloc_data()\n");
        }
        error_code = MPI_ERR_UNKNOWN;
        goto done;
    }
    
    double start_time = MPI_Wtime();
    error_code = solve(&run_config);
    CHECK_ERROR_CODE(error_code);
    double end_time = MPI_Wtime();
    
    double proc_time = end_time - start_time;
    double elapsed_time;
    MPI_Reduce(&proc_time, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, run_config.mpi_config->comm);
    if (run_config.mpi_config->proc_id == 0) {
        printf("elapsed time: %lf s\n\n", elapsed_time);
    }
    
    error_code = MPI_SUCCESS;
done:
    free_data(&run_config);
    MPI_Finalize();
    return error_code ? 1 : 0;
}
