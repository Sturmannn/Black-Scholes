#include <cmath>
#include <iostream>
#include <cstdlib>
#include <omp.h>

struct Option
{
  float* s0; // цена акции в начальное время
  float* T;  // время исполнения опциона в годах
  float* K;  // страйк
  float* C;  // Справедливая цена опциона
};

const float sig = 0.2f; // волатильность
const float r = 0.05f; // процентная ставка

float start, finish; // замеры времени (засечки)
float dt; // время работы блока кода (изменение времени)
const int N = 50000000; // количество опционов для подсчёта

const float invsqrt2 = std::sqrt(2.0f); // инварианты
const float inv_square_sig = sig * sig;


void AllocMemForStruct(Option& opt)
{
  opt.C = new float[N];
  opt.K = new float[N];
  opt.T = new float[N];
  opt.s0 = new float[N];
}
void DeteleStruct(Option& opt)
{
  delete[] opt.C;
  delete[] opt.K;
  delete[] opt.T;
  delete[] opt.s0;
}

void GetOptionPrices(Option& opt)
{
  float d1, d2, erf1, erf2;
  #pragma omp parallel private(d1,d2,erf1,erf2)
  {
    #pragma omp for
    for (int i = 0; i < N; i++)
    {
      d1 = (std::log(opt.s0[i] / opt.K[i]) + (r + inv_square_sig / 2) * opt.T[i]) / (sig * std::sqrt(opt.T[i]));
      d2 = (std::log(opt.s0[i] / opt.K[i]) + (r - inv_square_sig / 2) * opt.T[i]) / (sig * std::sqrt(opt.T[i]));
      erf1 = 0.5f + std::erf(d1 / invsqrt2) * 0.5f;
      erf2 = 0.5f + std::erf(d2 / invsqrt2) * 0.5f;

      opt.C[i] = opt.s0[i] * erf1 - opt.K[i] * std::exp((-1.0f) * r * opt.T[i]) * erf2;
    }
  }
}

int main(int argc, char* argv[])
{
  omp_set_num_threads(8);
  srand(5);

  Option sample;
  AllocMemForStruct(sample);

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    sample.K[i] = (float)rand() / (float)RAND_MAX * (250.0f - 50.0f) + 50.0f;
    sample.s0[i] = (float)rand() / (float)RAND_MAX * (150.0f - 50.0f) + 50.0f; // Случайные числа в диапазоне
    sample.T[i] = (float)rand() / (float)RAND_MAX * (5.0f - 1.0f) + 1.0f;
  }

  start = (float)omp_get_wtime();
  GetOptionPrices(sample);
  finish = (float)omp_get_wtime();
  dt = (float)finish - (float)start;

  //for (int i = 0; i < 3; i++)
  //{
  //  std::cout << "C =  " << sample.C[i] << std::endl;
  //  std::cout << "K =  " << sample.K[i] << std::endl;
  //  std::cout << "s0 =  " << sample.s0[i] << std::endl;
  //  std::cout << "T =  " << sample.T[i] << std::endl;
  //  std::cout << std::endl;
  //}

  std::cout << "dt = " << dt << std::endl;
  system("pause");
  DeteleStruct(sample);
  return 0;
}

