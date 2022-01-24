#include "solve.h"

#include <stdio.h>
#include <string.h>
#include <math.h>


int sync_borders(const struct RunConfig *run_config, double *a)
{
    int error_code =  MPI_ERR_UNKNOWN;
    void *sendbuf;
    void *recvbuf;
    int count;
    
    // Отправка вправо
    sendbuf = &a[run_config->domain_m * (run_config->domain_n + 2)];
    recvbuf = &a[                   0 * (run_config->domain_n + 2)];
    count = run_config->domain_n + 2;
    error_code = MPI_Sendrecv(sendbuf, count, MPI_DOUBLE, run_config->mpi_config->right_x_id, 0,
                              recvbuf, count, MPI_DOUBLE, run_config->mpi_config->left_x_id, 0,
                              run_config->mpi_config->comm, MPI_STATUS_IGNORE);
    CHECK_ERROR_CODE(error_code);
    
    // Отправка влево
    sendbuf = &a[                         1 * (run_config->domain_n + 2)];
    recvbuf = &a[(run_config->domain_m + 1) * (run_config->domain_n + 2)];
    count = run_config->domain_n + 2;
    error_code = MPI_Sendrecv(sendbuf, count, MPI_DOUBLE, run_config->mpi_config->left_x_id, 0,
                              recvbuf, count, MPI_DOUBLE, run_config->mpi_config->right_x_id, 0,
                              run_config->mpi_config->comm, MPI_STATUS_IGNORE);
    CHECK_ERROR_CODE(error_code);
    
    // Отправка вниз и вверх
    count = run_config->domain_m + 2;
    for (my_int i = 0; i <= run_config->domain_m + 1; ++i) {
        run_config->buf0[i] = a[i * (run_config->domain_n + 2) + 1];
        run_config->buf1[i] = a[i * (run_config->domain_n + 2) + run_config->domain_n];
    }
    error_code = MPI_Sendrecv_replace(run_config->buf0, count, MPI_DOUBLE, run_config->mpi_config->bottom_y_id, 0,
                                      run_config->mpi_config->top_y_id, 0,
                                      run_config->mpi_config->comm, MPI_STATUS_IGNORE);
    CHECK_ERROR_CODE(error_code);
    error_code = MPI_Sendrecv_replace(run_config->buf1, count, MPI_DOUBLE, run_config->mpi_config->top_y_id, 0,
                                      run_config->mpi_config->bottom_y_id, 0,
                                      run_config->mpi_config->comm, MPI_STATUS_IGNORE);
    CHECK_ERROR_CODE(error_code);
    for (my_int i = 0; i <= run_config->domain_m + 1; ++i) {
        a[i * (run_config->domain_n + 2) + 0] = run_config->buf1[i];
        a[i * (run_config->domain_n + 2) + run_config->domain_n + 1] = run_config->buf0[i];
    }
    
    error_code = MPI_SUCCESS;
done:
    return error_code;
}

double sync_error_value(const struct RunConfig *run_config, double part_error)
{
    double error_value;
    MPI_Allreduce(&part_error, &error_value, 1, MPI_DOUBLE, MPI_SUM, run_config->mpi_config->comm);
    error_value *= run_config->h1 * run_config->h2;
    error_value = sqrt(error_value);
    return error_value;
}

int solve(struct RunConfig *run_config)
{
    int error_code =  MPI_ERR_UNKNOWN;
    
    init_data(run_config);
    
    double error_value = 0;
    int iters_done = 0;
    double num = 0;
    double div = 0;
    double sb[2];
    double rb[2];
    double start_time = MPI_Wtime();
    do {
        // Ссылка для удобства обращения при промежуточных вычислениях
        double *a_residual = run_config->next_w;
        
        // Вычисляем невязку
        calc_aw_b(run_config, run_config->cur_w, run_config->residual, 1);
        
        // Синхронизация внешних границ невязки между процессами
        error_code = sync_borders(run_config, run_config->residual);
        CHECK_ERROR_CODE(error_code);
        
        // Умножаем на A невязку
        calc_aw_b(run_config, run_config->residual, a_residual, 0);
        
        // Находим частичные суммы скалярного произведения и нормы
        num = calc_num_part(run_config, a_residual);
        div = calc_div_part(run_config, a_residual);
        
        // Синхронизация num, div
        sb[0] = num;
        sb[1] = div;
        MPI_Allreduce(sb, rb, 2, MPI_DOUBLE, MPI_SUM, run_config->mpi_config->comm);
        num = rb[0];
        div = rb[1];
        
        // Находим итерационный параметр
        double tau = num / div;
        
        // Находим новые значения сеточной функции
        update_w(run_config, tau);
        
        // Синхронизация границ нового w
        error_code = sync_borders(run_config, run_config->next_w);
        CHECK_ERROR_CODE(error_code);
        
        // Вычисляем частичную сумму ошибки
        error_value = calc_partial_error(run_config);
        
        // Cинхронизация значения ошибки
        sb[0] = error_value;
        MPI_Allreduce(sb, rb, 1, MPI_DOUBLE, MPI_SUM, run_config->mpi_config->comm);
        error_value = rb[0];
        error_value *= run_config->h1 * run_config->h2;
        error_value = sqrt(error_value);
        
        // Записываем новое w в качестве текущего
        memcpy(run_config->cur_w, run_config->next_w, (run_config->domain_m + 2) * (run_config->domain_n + 2) * sizeof(run_config->cur_w[0]));
        
        ++iters_done;
        if (iters_done % 100 == 0 || iters_done == 1) {
            double part_error = calc_current_error_part(run_config);
            double cur_error = sync_error_value(run_config, part_error);
            double end_time = MPI_Wtime();
            double proc_time = end_time - start_time;
            start_time = end_time;
            double elapsed_time;
            MPI_Reduce(&proc_time, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, run_config->mpi_config->comm);
            if (run_config->mpi_config->proc_id == 0) {
                printf("iters_done == %d, iter_error == %.8lf, current error == %.8lf, chunk time == %.2lfs\n", iters_done, error_value, cur_error, elapsed_time);
            }
        }
    } while (error_value >= run_config->eps && iters_done);
    
    // Можно собрать значение w на нулевом процессе, вывести вперемешку или просто найти ошибку от точного решения
    error_value = calc_current_error_part(run_config);
    error_value = sync_error_value(run_config, error_value);
    if (run_config->mpi_config->proc_id == 0) {
        printf("final error == %lg, iters_done == %d\n", error_value, iters_done);
    }
    
#if 0
    (i, j)
    my_int start_i = 1;
    my_int start_j = 1;
    my_int stop_i = run_config->domain_m;
    my_int stop_j = run_config->domain_n;
    for (my_int i = start_i; i <= stop_i; ++i) {
        for (my_int j = start_j; j <= stop_j; ++j) {
            my_int idx = i * (run_config->domain_n + 2) + j;
            printf("%d %d %.4lf\n", run_config->start_i + i - 1, run_config->start_j + j - 1, run_config->cur_w[idx]);
        }
    }
#endif
    
    error_code = MPI_SUCCESS;
done:
    return error_code;
}
