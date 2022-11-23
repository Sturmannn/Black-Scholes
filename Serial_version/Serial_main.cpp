#include <time.h>
#include <math.h>
#include <iostream>

struct Option
{
  float s0 = 100.0f; // цена акции в начальное время
  float T = 3.0f; // время исполнения опциона в годах
  float K = 100.0f; // Страйк
  float C; // Справедливая цена опциона
};

const float sig = 0.2f; // волатильность
const float r = 0.05f; // процентная ставка

clock_t start, finish;// замеры времени (засечки)
float dt; // время работы блока кода (изменение времени)
const int N = 20000000; // количество опционов для подсчёта

void GetOptionPrices(Option* opt)
{
  float d1, d2, erf1, erf2;
  for (int i = 0; i < N; i++)
  {
    opt[i].K = (float)rand() / (float)RAND_MAX * (250.0f - 50.0f) + 50.0f;
    opt[i].s0 = (float)rand() / (float)RAND_MAX * (150.0f - 50.0f) + 50.0f; // Случайные числа в диапазоне
    opt[i].T = (float)rand() / (float)RAND_MAX * (5.0f - 1.0f) + 1.0f;
    d1 = log(opt[i].s0 / opt[i].K) + ((r + (sig * sig / 2.0f) * opt[i].T) / (sig * sqrt(opt[i].T)));
    d2 = log(opt[i].s0 / opt[i].K) + ((r - (sig * sig / 2.0f) * opt[i].T) / (sig * sqrt(opt[i].T)));
    erf1 = 0.5f + erf(d1 / sqrt(2.0f)) / 2.0f;
    erf2 = 0.5f + erf(d2 / sqrt(2.0f)) / 2.0f;
    opt[i].C = opt[i].s0 * erf1 - opt[i].K * exp(-1.0f * r * opt[i].T) * erf2;
  }
}

int main(int argc, char* argv[])
{
  Option* sample = new Option[N];

  start = clock();
  GetOptionPrices(sample);
  finish = clock();
  dt = ((float)finish - (float)start) / (float)CLOCKS_PER_SEC;

  //for (int i = 0; i < 5; i++)
  //{
  //  std::cout << "C =  " << sample[i].C << std::endl;
  //  std::cout << "K =  " << sample[i].K << std::endl;
  //  std::cout << "s0 =  " << sample[i].s0 << std::endl;
  //  std::cout << "T =  " << sample[i].T << std::endl;
  //  std::cout << std::endl;
  //}

  std::cout << "dt = " << dt << std::endl;
  delete[] sample;

  return 0;
}
