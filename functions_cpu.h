#pragma once

#include "common.h"


double q(double x, double y);

double F(double x, double y);

double F_border(const struct RunConfig *run_config, my_int i, my_int j, double x, double y);
