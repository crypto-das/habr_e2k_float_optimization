#pragma once

#include "common.h"


int alloc_data(struct RunConfig *run_config); 

void free_data(struct RunConfig *run_config);

void init_data(const struct RunConfig *run_config);

void calc_aw_b(const struct RunConfig *run_config, const double *src, double *dst, double alpha);

void calc_tau_part(const struct RunConfig *run_config, const double *a_residual, double *num_out, double *div_out);

double update_w_calc_partial_error(const struct RunConfig *run_config, double tau);

double calc_current_error_part(const struct RunConfig *run_config);
