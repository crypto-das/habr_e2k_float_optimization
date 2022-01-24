#include "functions_cpu.h"

#include <math.h>

#include "common.h"


double q(double x, double y)
{
    return x + y;
}

double F(double x, double y)
{
    double xpy = x + y;
    double x2 = x * x;
    double y2 = y * y;
    double xyp4 = x * y + 4;
    double sqr = sqrt(xyp4);
    return (x2 + y2) / (4 * xyp4 * sqr) + xpy * sqr;
}

double F_border(const struct RunConfig *run_config, my_int i, my_int j, double x, double y)
{
    double xpy = x + y;
    double x2 = x * x;
    double y2 = y * y;
    double xyp4 = x * y + 4;
    double sqr = sqrt(xyp4);
    double res = (x2 + y2) / (4 * xyp4 * sqr) + xpy * sqr;
    if (run_config->start_i == 0 && i == 1) {
        double psy_l = sqr - y / (2 * sqr);
        res += 2 * run_config->inv_h1 * psy_l;
    }
    if (run_config->stop_i == run_config->m - 1 && i == run_config->domain_m) {
        double psy_r = sqr + y / (2 * sqr);
        res += 2 * run_config->inv_h1 * psy_r;
    }
    if (run_config->start_j == 0 && j == 1) {
        double psy_b = sqr - x / (2 * sqr);
        res += 2 * run_config->inv_h2 * psy_b;
    }
    if (run_config->stop_j == run_config->n - 1 && j == run_config->domain_n) {
        double psy_t = sqr + x / (2 * sqr);
        res += 2 * run_config->inv_h2 * psy_t;
    }
    return res;
}
