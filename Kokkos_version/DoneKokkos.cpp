#include <cmath>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
//#include <Kokkos_Complex.hpp>
#include <fstream>

struct Option
{
  Kokkos::View<float*> s0; // цена акции в начальное время
  Kokkos::View<float*> T;  // время исполнения опциона в годах
  Kokkos::View<float*> K;  // страйк
  Kokkos::View<float*> C;  // Справедливая цена опциона
};

const float sig = 0.2f; // волатильность
const float r = 0.05f; // процентная ставка

float start, finish; // замеры времени (засечки)
float dt; // время работы блока кода (изменение времени)
const int N = 50000000; // количество опционов для подсчёта

const float invsqrt2 = Kokkos::sqrt(2.0f); // инварианты
const float inv_square_sig = sig * sig;


void Set_Count_Of_Options(Option& opt)
{
  Kokkos::resize(Kokkos::WithoutInitializing, opt.C, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.K, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.T, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.s0, N);
}

void GetOptionPrices(Option& opt)
{
  Kokkos::parallel_for("computing_opt", N, KOKKOS_LAMBDA(const int& i)
  {
    float d1, d2, erf1, erf2;
    d1 = (Kokkos::log(opt.s0(i) / opt.K(i)) + (r + inv_square_sig / 2) * opt.T(i)) / (sig * Kokkos::sqrt(opt.T(i)));
    d2 = (Kokkos::log(opt.s0(i) / opt.K(i)) + (r - inv_square_sig / 2) * opt.T(i)) / (sig * Kokkos::sqrt(opt.T(i)));
    erf1 = 0.5f + Kokkos::erf(d1 / invsqrt2) * 0.5f;
    erf2 = 0.5f + Kokkos::erf(d2 / invsqrt2) * 0.5f;

    opt.C(i) = opt.s0(i) * erf1 - opt.K(i) * std::exp((-1.0f) * r * opt.T(i)) * erf2;
  });
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(
    Kokkos::InitializationSettings()
    .set_disable_warnings(false)
    .set_num_threads(2));
  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);
  //Kokkos::initialize(argc, argv);
  //srand(5);

  Kokkos::Random_XorShift64_Pool<> random_pool(5);

  Option sample;
  Set_Count_Of_Options(sample);

  Kokkos::parallel_for("computing_opt", N, KOKKOS_LAMBDA(const int& i)
  {
    auto generator = random_pool.get_state();

    sample.K(i) = (float)generator.rand() / (float)generator.MAX_RAND * (250.0f - 50.0f) + 50.0f;
    sample.s0(i) = (float)generator.rand() / (float)generator.MAX_RAND * (150.0f - 50.0f) + 50.0f; // Случайные числа в диапазоне
    sample.T(i) = (float)generator.rand() / (float)generator.MAX_RAND * (5.0f - 1.0f) + 1.0f;
    random_pool.free_state(generator);
  });

  Kokkos::Timer timer;
  float start = (float)timer.seconds();
  timer.reset();

  GetOptionPrices(sample);

  float finish = (float)timer.seconds();
  //dt = (float)finish - (float)start;

  //std::ofstream out("./Zameri.txt", std::ios::app);
  //if (out.is_open())
  //{
  //  out << dt << std::endl;
  //}
  //else std::cout << "File can't open!";
  //out.close();
  std::cout << "ITS OKAY: " << finish << "\n";
  Kokkos::finalize();

  return 0;
}

