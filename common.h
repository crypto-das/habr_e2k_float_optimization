#pragma once

#include <stdio.h>

#define CHECK_ERROR_CODE(error_code) \
    if (error_code) { \
        printf("File: %s, Line: %d, error code: %d\n", __FILE__, __LINE__, error_code); \
        goto done; \
    }

#define DEBUG_POINT \
    printf("File: %s, Line: %d, pid: %d\n", __FILE__, __LINE__, run_config->mpi_config->proc_id);


#define my_int ssize_t


struct MpiConfig;

struct RunConfig
{
    struct MpiConfig *mpi_config;
    
    my_int m;
    my_int n;
    my_int domain_m;
    my_int domain_n;
    double h1;
    double h2;
    double inv_h1;
    double inv_h2;
    double sqinv_h1;
    double sqinv_h2;
    my_int start_i;
    my_int start_j;
    my_int stop_i;
    my_int stop_j;
    double eps;
    
    double *cur_w;
    double *residual;
    double *next_w;
    double *b_mat;
    double *q_mat;
    double *buf0;
    double *buf1;
};
