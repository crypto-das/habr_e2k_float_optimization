#pragma once

#include "common.h"


int alloc_data(struct RunConfig *run_config); 

void free_data(struct RunConfig *run_config);

void init_data(const struct RunConfig *run_config);

void calc_aw_b(const struct RunConfig *run_config, const double *src, double *dst, double alpha);

double calc_num_part(const struct RunConfig *run_config, const double *a_residual);

double calc_div_part(const struct RunConfig *run_config, const double *a_residual);

void update_w(const struct RunConfig *run_config, double tau);

double calc_partial_error(const struct RunConfig *run_config);

double calc_current_error_part(const struct RunConfig *run_config);
