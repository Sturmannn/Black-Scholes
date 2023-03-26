//#include <cmath>
//#include <iostream>
//#include <cstdlib>
//#include <omp.h>
//
//
//struct Option
//{
//  float s0 = 0.0f; // цена акции в начальное время
//  float T = 0.0f; // время исполнения опциона в годах
//  float K = 0.0f; // страйк
//  float C = 0.0f; // Справедливая цена опциона
//};
//
//const float sig = 0.2f; // волатильность
//const float r = 0.05f; // процентная ставка
//
//float start, finish; // замеры времени (засечки)
//float dt; // время работы блока кода (изменение времени)
//const int N = 50000000; // количество опционов для подсчёта
//
//const float invsqrt2 = std::sqrt(2.0f); // инварианты
//const float inv_square_sig = sig * sig;
//
//void GetOptionPrices(Option* opt)
//{
//  float d1, d2, erf1, erf2;
//  #pragma omp parallel private(d1,d2,erf1,erf2)
//  {
//    //#pragma omp master
//    //{
//    //  std::cout << "num threads " << omp_get_num_threads() << std::endl;
//    //}
//    #pragma omp for
//    for (int i = 0; i < N; i++)
//    {
//      d1 = (std::log(opt[i].s0 / opt[i].K) + (r + inv_square_sig / 2) * opt[i].T) / (sig * std::sqrt(opt[i].T));
//      d2 = (std::log(opt[i].s0 / opt[i].K) + (r - inv_square_sig / 2) * opt[i].T) / (sig * std::sqrt(opt[i].T));
//      erf1 = 0.5f + std::erf(d1 / invsqrt2) * 0.5f;
//      erf2 = 0.5f + std::erf(d2 / invsqrt2) * 0.5f;
//
//      opt[i].C = opt[i].s0 * erf1 - opt[i].K * std::exp((-1.0f) * r * opt[i].T) * erf2;
//    }
//  }
//}
//
//int main(int argc, char* argv[])
//{
//  omp_set_num_threads(1);
//  srand(5);
//  Option* sample = new Option[N];
//  #pragma omp parallel for
//  for (int i = 0; i < N; i++)
//  {
//    sample[i].K = (float)rand() / (float)RAND_MAX * (250.0f - 50.0f) + 50.0f;
//    sample[i].s0 = (float)rand() / (float)RAND_MAX * (150.0f - 50.0f) + 50.0f; // Случайные числа в диапазоне
//    sample[i].T = (float)rand() / (float)RAND_MAX * (5.0f - 1.0f) + 1.0f;
//  }
//
//  start = (float)omp_get_wtime();
//  GetOptionPrices(sample);
//  finish = (float)omp_get_wtime();
//  dt = (float)finish - (float)start;
//
//  //for (int i = 0; i < 3; i++)
//  //{
//  //  std::cout << "C =  " << sample[i].C << std::endl;
//  //  std::cout << "K =  " << sample[i].K << std::endl;
//  //  std::cout << "s0 =  " << sample[i].s0 << std::endl;
//  //  std::cout << "T =  " << sample[i].T << std::endl;
//  //  std::cout << std::endl;
//  //}
//
//  std::cout << "dt = " << dt << std::endl;
//  delete[] sample;
//  return 0;
//}
//
