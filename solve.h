#pragma once

#include <mpi.h>

#include "common.h"
#include "solve_cpu.h"


struct MpiConfig
{
    MPI_Comm comm;
    int num_procs;
    int proc_id;
    int dims[2];
    int proc_coords[2];
    int left_x_id;
    int right_x_id;
    int bottom_y_id;
    int top_y_id;
};


int solve(struct RunConfig *run_config);
