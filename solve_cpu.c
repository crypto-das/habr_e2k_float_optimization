#include "solve_cpu.h"

#include <stdlib.h>
#include <math.h>

#if defined(__e2k__)
#include <e2kintrin.h>
#endif


int alloc_data(struct RunConfig *run_config)
{
    int error_code = EXIT_FAILURE;
    
    my_int domain_size = (run_config->domain_m + 2) * (run_config->domain_n + 2);
    run_config->cur_w = aligned_alloc(16, domain_size * sizeof(*run_config->cur_w));
    if (!run_config->cur_w) {
        error_code = EXIT_FAILURE;
        goto done;
    }
    
    run_config->residual = aligned_alloc(16, domain_size * sizeof(*run_config->residual));
    if (!run_config->residual) {
        error_code = EXIT_FAILURE;
        goto done;
    }
    
    run_config->next_w = aligned_alloc(16, domain_size * sizeof(*run_config->next_w));
    if (!run_config->residual) {
        error_code = EXIT_FAILURE;
        goto done;
    }
    
    run_config->b_mat = aligned_alloc(16, domain_size * sizeof(*run_config->b_mat));
    if (!run_config->b_mat) {
        error_code = EXIT_FAILURE;
        goto done;
    }
    
    run_config->q_mat = aligned_alloc(16, domain_size * sizeof(*run_config->q_mat));
    if (!run_config->q_mat) {
        error_code = EXIT_FAILURE;
        goto done;
    }
    
    run_config->buf0 = aligned_alloc(16, (run_config->domain_m + 2) * sizeof(*run_config->buf0));
    if (!run_config->buf0) {
        error_code = EXIT_FAILURE;
        goto done;
    }
    
    run_config->buf1 = aligned_alloc(16, (run_config->domain_m + 2) * sizeof(*run_config->buf1));
    if (!run_config->buf1) {
        error_code = EXIT_FAILURE;
        goto done;
    }
    
    error_code = 0;
done:
    return error_code;
}

void free_data(struct RunConfig *run_config)
{
    free(run_config->cur_w);
    free(run_config->residual);
    free(run_config->next_w);
    free(run_config->b_mat);
    free(run_config->q_mat);
    free(run_config->buf0);
    free(run_config->buf1);
    run_config->cur_w = NULL;
    run_config->residual = NULL;
    run_config->next_w = NULL;
    run_config->b_mat = NULL;
    run_config->q_mat = NULL;
    run_config->buf0 = NULL;
    run_config->buf1 = NULL;
}

void init_data(const struct RunConfig *run_config)
{
    my_int start_i = 1;
    my_int start_j = 1;
    my_int stop_i = run_config->domain_m;
    my_int stop_j = run_config->domain_n;
    
    // Возможна инициализация точным решением для проверки
    for (my_int i = 0; i <= run_config->domain_m + 1; ++i) {
        for (my_int j = 0; j <= run_config->domain_n + 1; ++j) {
            my_int idx = i * (run_config->domain_n + 2) + j;
#if 0
            double x = run_config->h1 * (run_config->start_i + i - 1);
            double y = run_config->h2 * (run_config->start_j + j - 1);
            double xyp4 = x * y + 4;
            double sqr = sqrt(xyp4);
            run_config->cur_w[idx] = sqr;
#endif
            run_config->cur_w[idx] = 0;
        }
    }
    
    // (i, j)
    for (my_int i = start_i; i <= stop_i; ++i) {
        for (my_int j = start_j; j <= stop_j; ++j) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            double x = run_config->h1 * (run_config->start_i + i - 1);
            double y = run_config->h2 * (run_config->start_j + j - 1);
            double xpy = x + y;
            run_config->q_mat[idx] = xpy;
            double x2 = x * x;
            double y2 = y * y;
            double xyp4 = x * y + 4;
            double sqr = sqrt(xyp4);
            run_config->b_mat[idx] = (x2 + y2) / (4 * xyp4 * sqr) + xpy * sqr;
            if (run_config->start_i == 0 && i == start_i) {
                double psy_l = sqr - y / (2 * sqr);
                run_config->b_mat[idx] += 2 * run_config->inv_h1 * psy_l;
            }
            if (run_config->stop_i == run_config->m - 1 && i == stop_i) {
                double psy_r = sqr + y / (2 * sqr);
                run_config->b_mat[idx] += 2 * run_config->inv_h1 * psy_r;
            }
            if (run_config->start_j == 0 && j == start_j) {
                double psy_b = sqr - x / (2 * sqr);
                run_config->b_mat[idx] += 2 * run_config->inv_h2 * psy_b;
            }
            if (run_config->stop_j == run_config->n - 1 && j == stop_j) {
                double psy_t = sqr + x / (2 * sqr);
                run_config->b_mat[idx] += 2 * run_config->inv_h2 * psy_t;
            }
        }
    }
}

void calc_aw_b(const struct RunConfig *run_config, const double * restrict src, double * restrict dst, double alpha)
{
    double * restrict b_mat = run_config->b_mat;
    double * restrict q_mat = run_config->q_mat;
    
    my_int start_i = 1;
    my_int start_j = 1;
    my_int stop_i = run_config->domain_m;
    my_int stop_j = run_config->domain_n;
    if (run_config->start_i == 0) {
        ++start_i;
        // (1, j)
        for (my_int j = 1; j <= run_config->domain_n; ++j) {
            my_int idx = 1 * (run_config->domain_n + 2) + j;
            double w0j  = src[idx];
            double w1j  = src[2 * (run_config->domain_n + 2) + j];
            double w0jp = src[1 * (run_config->domain_n + 2) + j + 1];
            double w0jm = src[1 * (run_config->domain_n + 2) + j - 1];
            dst[idx] = 2 * run_config->sqinv_h1 * (w0j - w1j) + (q_mat[idx] + 2 * run_config->inv_h1) * w0j - (w0jp + w0jm - 2 * w0j) * run_config->sqinv_h2 - alpha * b_mat[idx];
        }
    }
    if (run_config->stop_i == run_config->m - 1) {
        --stop_i;
        // (M, j)
        for (my_int j = 1; j <= run_config->domain_n; ++j) {
            my_int idx = run_config->domain_m * (run_config->domain_n + 2) + j;
            double wmj  = src[idx];
            double wmmj = src[(run_config->domain_m - 1) * (run_config->domain_n + 2) + j];
            double wmjp = src[run_config->domain_m       * (run_config->domain_n + 2) + j + 1];
            double wmjm = src[run_config->domain_m       * (run_config->domain_n + 2) + j - 1];
            dst[idx] = 2 * run_config->sqinv_h1 * (wmj - wmmj) + (q_mat[idx] + 2 * run_config->inv_h1) * wmj - (wmjp + wmjm - 2 * wmj) * run_config->sqinv_h2 - alpha * b_mat[idx];
        }
    }
    if (run_config->start_j == 0) {
        ++start_j;
        // (i, 1)
        for (my_int i = 1; i <= run_config->domain_m; ++i) {
            my_int idx = i * (run_config->domain_n + 2) + 1;
            double wi0  = src[idx];
            double wi1  = src[i       * (run_config->domain_n + 2) + 2];
            double wip0 = src[(i + 1) * (run_config->domain_n + 2) + 1];
            double wim0 = src[(i - 1) * (run_config->domain_n + 2) + 1];
            dst[idx] = 2 * run_config->sqinv_h2 * (wi0 - wi1) + (q_mat[idx] + 2 * run_config->inv_h2) * wi0 - (wip0 + wim0 - 2 * wi0) * run_config->sqinv_h1 - alpha * b_mat[idx];
        }
    }
    if (run_config->stop_j == run_config->n - 1) {
        --stop_j;
        // (i, N)
        for (my_int i = 1; i <= run_config->domain_m; ++i) {
            my_int idx = i * (run_config->domain_n + 2) + run_config->domain_n;
            double win  = src[idx];
            double winm = src[i       * (run_config->domain_n + 2) + run_config->domain_n - 1];
            double wipn = src[(i + 1) * (run_config->domain_n + 2) + run_config->domain_n];
            double wimn = src[(i - 1) * (run_config->domain_n + 2) + run_config->domain_n];
            dst[idx] = 2 * run_config->sqinv_h2 * (win - winm) + (q_mat[idx] + 2 * run_config->inv_h2) * win - (wipn + wimn - 2 * win) * run_config->sqinv_h1 - alpha * b_mat[idx];
        }
    }
    // (i, j)
#if defined(__e2k__)
    my_int head = start_j % 2;
    start_j += head;
    my_int tail = (stop_j - start_j + 1) % 2;
    stop_j -= tail;
    
    type_union_128 sqinv_h1;
    sqinv_h1.d.d0 = run_config->sqinv_h1;
    sqinv_h1.d.d1 = run_config->sqinv_h1;
    
    type_union_128 sqinv_h2;
    sqinv_h2.d.d0 = run_config->sqinv_h2;
    sqinv_h2.d.d1 = run_config->sqinv_h2;
    
    type_union_128 const_2;
    const_2.d.d0 = 2.0;
    const_2.d.d1 = 2.0;
    
    type_union_128 v2alpha;
    v2alpha.d.d0 = alpha;
    v2alpha.d.d1 = alpha;
    
    for (my_int i = start_i; i <= stop_i; ++i) {
        if (head) {
            my_int j = 1;
            my_int idx = i * (run_config->domain_n + 2) + j;
            double wij  = src[idx];
            double wipj = src[(i + 1) * (run_config->domain_n + 2) + j];
            double wimj = src[(i - 1) * (run_config->domain_n + 2) + j];
            double wijp = src[i       * (run_config->domain_n + 2) + j + 1];
            double wijm = src[i       * (run_config->domain_n + 2) + j - 1];
            double laplacian = (wipj + wimj - 2 * wij) * run_config->sqinv_h1 + (wijp + wijm - 2 * wij) * run_config->sqinv_h2;
            dst[idx] = q_mat[idx] * wij - laplacian - alpha * b_mat[idx];
        }
        
        const __v2di mix_doubles = { 0x0f0e0d0c0b0a0908, 0x1716151413121110 };
        __v2di src_0 = *((__v2di *) &src[i * (run_config->domain_n + 2) + start_j - 2]);
        __v2di src_1 = *((__v2di *) &src[i * (run_config->domain_n + 2) + start_j]);
        __v2di wijm = __builtin_e2k_qppermb(src_1, src_0, mix_doubles);
        
        for (my_int j = start_j; j < stop_j; j += 2) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            __v2di wij = *((__v2di *) &src[idx]);
            __v2di wipj = *((__v2di *) &src[(i + 1) * (run_config->domain_n + 2) + j]);
            __v2di wimj = *((__v2di *) &src[(i - 1) * (run_config->domain_n + 2) + j]);
            
            src_0 = wij;
            src_1 = *((__v2di *) &src[idx + 2]);
            __v2di wijp = __builtin_e2k_qppermb(src_1, src_0, mix_doubles);
            
            __v2di t0 = __builtin_e2k_qpfaddd(wipj, wimj);
            __v2di t1 = __builtin_e2k_qpfmuld(const_2.__v2di, wij);
            __v2di t2 = __builtin_e2k_qpfaddd(wijp, wijm);
            __v2di t3 = __builtin_e2k_qpfsubd(t0, t1);
            __v2di t4 = __builtin_e2k_qpfsubd(t2, t1);
            __v2di t5 = __builtin_e2k_qpfmuld(t3, sqinv_h1.__v2di);
            __v2di t6 = __builtin_e2k_qpfmuld(t4, sqinv_h2.__v2di);
            __v2di laplacian = __builtin_e2k_qpfaddd(t5, t6);
            __v2di t7 = __builtin_e2k_qpfmuld(wij, *((__v2di *) &q_mat[idx]));
            __v2di t8 = __builtin_e2k_qpfmuld(v2alpha.__v2di, *((__v2di *) &b_mat[idx]));
            *((__v2di *) &dst[idx]) = __builtin_e2k_qpfsubd(__builtin_e2k_qpfsubd(t7, laplacian), t8);
            wijm = wijp;
        }
        
        if (tail) {
            my_int j = stop_j + 1;
            my_int idx = i * (run_config->domain_n + 2) + j;
            double wij  = src[idx];
            double wipj = src[(i + 1) * (run_config->domain_n + 2) + j];
            double wimj = src[(i - 1) * (run_config->domain_n + 2) + j];
            double wijp = src[i       * (run_config->domain_n + 2) + j + 1];
            double wijm = src[i       * (run_config->domain_n + 2) + j - 1];
            double laplacian = (wipj + wimj - 2 * wij) * run_config->sqinv_h1 + (wijp + wijm - 2 * wij) * run_config->sqinv_h2;
            dst[idx] = q_mat[idx] * wij - laplacian - alpha * b_mat[idx];
        }
    }
#else
    for (my_int i = start_i; i <= stop_i; ++i) {
        for (my_int j = start_j; j <= stop_j; ++j) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            double wij  = src[idx];
            double wipj = src[(i + 1) * (run_config->domain_n + 2) + j];
            double wimj = src[(i - 1) * (run_config->domain_n + 2) + j];
            double wijp = src[i       * (run_config->domain_n + 2) + j + 1];
            double wijm = src[i       * (run_config->domain_n + 2) + j - 1];
            double laplacian = (wipj + wimj - 2 * wij) * run_config->sqinv_h1 + (wijp + wijm - 2 * wij) * run_config->sqinv_h2;
            dst[idx] = q_mat[idx] * wij - laplacian - alpha * b_mat[idx];
        }
    }
#endif
    // Угловые точки заполняем строго в конце
    if (run_config->start_i == 0) {
        // (1, 1)
        if (run_config->start_j == 0) {
            my_int idx = 1 * (run_config->domain_n + 2) + 1;
            double w00  = src[idx];
            double w10  = src[2 * (run_config->domain_n + 2) + 1];
            double w01  = src[1 * (run_config->domain_n + 2) + 2];
            dst[idx] = 2 * run_config->sqinv_h1 * (w00 - w10) + 2 * run_config->sqinv_h2 * (w00 - w01) + (q_mat[idx] + 2 * (run_config->inv_h1 + run_config->inv_h2)) * w00 - alpha * b_mat[idx];
        }
        // (1, N)
        if (run_config->stop_j == run_config->n - 1) {
            my_int idx = 1 * (run_config->domain_n + 2) + run_config->domain_n;
            double w0n  = src[idx];
            double w1n  = src[2 * (run_config->domain_n + 2) + run_config->domain_n];
            double w0nm = src[1 * (run_config->domain_n + 2) + run_config->domain_n - 1];
            dst[idx] = 2 * run_config->sqinv_h1 * (w0n - w1n) + 2 * run_config->sqinv_h2 * (w0n - w0nm) + (q_mat[idx] + 2 * (run_config->inv_h1 + run_config->inv_h2)) * w0n - alpha * b_mat[idx];
        }
    }
    if (run_config->stop_i == run_config->m - 1) {
        // (M, 1)
        if (run_config->start_j == 0) {
            my_int idx = run_config->domain_m * (run_config->domain_n + 2) + 1;
            double wm0  = src[idx];
            double wmm0 = src[(run_config->domain_m - 1) * (run_config->domain_n + 2) + 1];
            double wm1  = src[run_config->domain_m       * (run_config->domain_n + 2) + 2];
            dst[idx] = 2 * run_config->sqinv_h1 * (wm0 - wmm0) + 2 * run_config->sqinv_h2 * (wm0 - wm1) + (q_mat[idx] + 2 * (run_config->inv_h1 + run_config->inv_h2)) * wm0 - alpha * b_mat[idx];
        }
        // (M, N)
        if (run_config->stop_j == run_config->n - 1) {
            my_int idx = run_config->domain_m * (run_config->domain_n + 2) + run_config->domain_n;
            double wmn  = src[idx];
            double wmmn = src[(run_config->domain_m - 1) * (run_config->domain_n + 2) + run_config->domain_n];
            double wmnm = src[run_config->domain_m       * (run_config->domain_n + 2) + run_config->domain_n - 1];
            dst[idx] = 2 * run_config->sqinv_h1 * (wmn - wmmn) + 2 * run_config->sqinv_h2 * (wmn - wmnm) + (q_mat[idx] + 2 * (run_config->inv_h1 + run_config->inv_h2)) * wmn - alpha * b_mat[idx];
        }
    }
}

void calc_tau_part(const struct RunConfig *run_config, const double * restrict a_residual, double * restrict num_out, double *div_out)
{
    double * restrict residual = run_config->residual;
    
    double num = 0;
    double div = 0;
    for (my_int i = 1; i <= run_config->domain_m; ++i) {
        double rhox = 1;
        if ((i == 1 && run_config->start_i == 0) || (i == run_config->domain_m && run_config->stop_i == run_config->m - 1)) {
            rhox = 0.5;
        }
        {
            my_int idx = i * (run_config->domain_n + 2) + 1;
            double rhoy = 1;
            if (run_config->start_j == 0) {
                rhoy = 0.5;
            }
            num += rhox * rhoy * a_residual[idx] * residual[idx];
            div += rhox * rhoy * a_residual[idx] * a_residual[idx];
        }
        {
            my_int idx = i * (run_config->domain_n + 2) + run_config->domain_n;
            double rhoy = 1;
            if (run_config->stop_j == run_config->n - 1) {
                rhoy = 0.5;
            }
            num += rhox * rhoy * a_residual[idx] * residual[idx];
            div += rhox * rhoy * a_residual[idx] * a_residual[idx];
        }
        
#if defined(__e2k__)
        type_union_128 tmp_num = { 0, 0 };
        type_union_128 tmp_div = { 0, 0 };
        my_int tail = (run_config->domain_n - 2) % 2;
        my_int stop_j = run_config->domain_n - 1 - tail;
        for (my_int j = 2; j < stop_j; j += 2) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            __v2di a_r = *((__v2di *) &a_residual[idx]);
            __v2di r = *((__v2di *) &residual[idx]);
            tmp_num.__v2di = __builtin_e2k_qpfaddd(tmp_num.__v2di, __builtin_e2k_qpfmuld(a_r, r));
            tmp_div.__v2di = __builtin_e2k_qpfaddd(tmp_div.__v2di, __builtin_e2k_qpfmuld(a_r, a_r));
        }
        num += (tmp_num.d.d0 + tmp_num.d.d1) * rhox;
        div += (tmp_div.d.d0 + tmp_div.d.d1) * rhox;
        
        if (tail) {
            my_int idx = i * (run_config->domain_n + 2) + stop_j + tail;
            num += a_residual[idx] * residual[idx] * rhox;
            div += a_residual[idx] * a_residual[idx] * rhox;
        }
#else
        double tmp_num = 0.0;
        double tmp_div = 0.0;
        for (my_int j = 2; j <= run_config->domain_n - 1; ++j) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            tmp_num += a_residual[idx] * residual[idx];
            tmp_div += a_residual[idx] * a_residual[idx];
        }
        num += tmp_num * rhox;
        div += tmp_div * rhox;
#endif
    }
    *num_out = num;
    *div_out = div;
}

double update_w_calc_partial_error(const struct RunConfig *run_config, double tau)
{
    double * restrict cur_w = run_config->cur_w;
    double * restrict next_w = run_config->next_w;
    double * restrict residual = run_config->residual;
    
#if defined(__e2k__)
#endif
    
    double error_value = 0;
    for (my_int i = 1; i <= run_config->domain_m; ++i) {
        double rhox = 1;
        if ((i == 1 && run_config->start_i == 0) || (i == run_config->domain_m && run_config->stop_i == run_config->m - 1)) {
            rhox = 0.5;
        }
        {
            my_int idx = i * (run_config->domain_n + 2) + 1;
            double rhoy = 1;
            if (run_config->start_j == 0) {
                rhoy = 0.5;
            }
            double diff = tau * residual[idx];
            next_w[idx] = cur_w[idx] - diff;
            error_value += rhox * rhoy * diff * diff;
        }
        {
            my_int idx = i * (run_config->domain_n + 2) + run_config->domain_n;
            double rhoy = 1;
            if (run_config->stop_j == run_config->n - 1) {
                rhoy = 0.5;
            }
            double diff = tau * residual[idx];
            next_w[idx] = cur_w[idx] - diff;
            error_value += rhox * rhoy * diff * diff;
        }
#if defined(__e2k__)
        type_union_128 v2tau;
        v2tau.d.d0 = tau;
        v2tau.d.d1 = tau;
        type_union_128 tmp_error_value = { 0, 0 };
        my_int tail = (run_config->domain_n - 2) % 2;
        my_int stop_j = run_config->domain_n - 1 - tail;
        for (my_int j = 2; j < stop_j; j += 2) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            __v2di diff = __builtin_e2k_qpfmuld(v2tau.__v2di, *((__v2di *) &residual[idx]));
            *((__v2di *) &next_w[idx]) = __builtin_e2k_qpfsubd(*((__v2di *) &cur_w[idx]), diff);
            tmp_error_value.__v2di = __builtin_e2k_qpfaddd(tmp_error_value.__v2di, __builtin_e2k_qpfmuld(diff, diff));
        }
        error_value += (tmp_error_value.d.d0 + tmp_error_value.d.d1) * rhox;
        
        if (tail) {
            my_int idx = i * (run_config->domain_n + 2) + stop_j + tail;
            double diff = tau * residual[idx];
            next_w[idx] = cur_w[idx] - diff;
            error_value += diff * diff * rhox;
        }
#else
        double tmp_error_value = 0.0;
        for (my_int j = 2; j <= run_config->domain_n - 1; ++j) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            double diff = tau * residual[idx];
            next_w[idx] = cur_w[idx] - diff;
            tmp_error_value += diff * diff;
        }
        error_value += tmp_error_value * rhox;
#endif
    }
    return error_value;
}

double calc_current_error_part(const struct RunConfig *run_config)
{
    double part_error = 0;
    for (my_int i = 1; i <= run_config->domain_m; ++i) {
        double rhox = 1;
        if ((i == 1 && run_config->start_i == 0) || (i == run_config->domain_m && run_config->stop_i == run_config->m - 1)) {
            rhox = 0.5;
        }
        for (my_int j = 1; j <= run_config->domain_n; ++j) {
            double rhoy = 1;
            if ((j == 1 && run_config->start_j == 0) || (j == run_config->domain_n && run_config->stop_j == run_config->n - 1)) {
                rhoy = 0.5;
            }
            double x = run_config->h1 * (run_config->start_i + i - 1);
            double y = run_config->h2 * (run_config->start_j + j - 1);
            double xy = x * y;
            double xyp4 = xy + 4;
            double u = sqrt(xyp4);
            my_int idx = i * (run_config->domain_n + 2) + j;
            double diff = fabs(u - run_config->cur_w[idx]);
            part_error += rhox * rhoy * diff * diff;
        }
    }
    return part_error;
}
