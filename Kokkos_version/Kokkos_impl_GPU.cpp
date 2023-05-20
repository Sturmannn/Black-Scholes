#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <fstream>



struct Option
{
  Kokkos::View<float*, Kokkos::SharedSpace> s0; // stock price at initial time
  Kokkos::View<float*, Kokkos::SharedSpace> T;  // option exercise time in years
  Kokkos::View<float*, Kokkos::SharedSpace> K;  // strike
  Kokkos::View<float*, Kokkos::SharedSpace> C;  // Fair option price
};

const float sig = 0.2f; // volatility
const float r = 0.05f; // interest rate
const int N = 50000000; // number of options to count

const float inv_square_sig = sig * sig;
//Kokkos::View<float*, Kokkos::SharedSpace> e("e", N);
//Kokkos::View<const float, Kokkos::CudaHostPinnedSpace> x("x");
const float invsqrt2 = 1.414213f;

void Set_Count_Of_Options(Option& opt)
{
  Kokkos::resize(Kokkos::WithoutInitializing, opt.C, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.K, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.T, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.s0, N);
}

void GetOptionPrices(Option& opt)
{
  Kokkos::parallel_for(
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
      Kokkos::DefaultExecutionSpace(), 0, N),
    KOKKOS_LAMBDA(const int& i)
  {
    float d1, d2, erf1, erf2;
    d1 = (Kokkos::log(opt.s0(i) / opt.K(i)) + (r + inv_square_sig / 2) * opt.T(i)) / (sig * Kokkos::sqrt(opt.T(i)));
    d2 = (Kokkos::log(opt.s0(i) / opt.K(i)) + (r - inv_square_sig / 2) * opt.T(i)) / (sig * Kokkos::sqrt(opt.T(i)));
    erf1 = 0.5f + Kokkos::erf(d1 / invsqrt2) * 0.5f;
    erf2 = 0.5f + Kokkos::erf(d2 / invsqrt2) * 0.5f;

    opt.C(i) = opt.s0(i) * erf1 - opt.K(i) * Kokkos::exp((-1.0f) * r * opt.T(i)) * erf2;
  });
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(
    Kokkos::InitializationSettings()
    .set_disable_warnings(false)
  );
  {
    Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

    Kokkos::Random_XorShift64_Pool<> random_pool(5);

    Option sample;
    Set_Count_Of_Options(sample);
    
    Kokkos::Timer timer;
    timer.reset();

    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
        Kokkos::DefaultExecutionSpace(), 0, N),
      KOKKOS_LAMBDA(const int& i)
    {
      auto generator = random_pool.get_state();

      //Random numbers in a range

      sample.K(i) = (float)generator.rand() / (float)generator.MAX_RAND * (250.0f - 50.0f) + 50.0f;
      sample.s0(i) = (float)generator.rand() / (float)generator.MAX_RAND * (150.0f - 50.0f) + 50.0f;
      sample.T(i) = (float)generator.rand() / (float)generator.MAX_RAND * (5.0f - 1.0f) + 1.0f;
      random_pool.free_state(generator);
    });

    Kokkos::fence(); //Synchronization

    float init_time = timer.seconds();

    timer.reset();

    GetOptionPrices(sample);

    Kokkos::fence();

    float exec_time = (float)timer.seconds();

    //=================================================================================================
    //Check correctness on the CPU
    float d1, d2, erf1, erf2, res;

    for (int i = 0; i < 3; i++)
    {
      d1 = (std::log(sample.s0[i] / sample.K[i]) + (r + inv_square_sig / 2) * sample.T[i]) / (sig * std::sqrt(sample.T[i]));
      d2 = (std::log(sample.s0[i] / sample.K[i]) + (r - inv_square_sig / 2) * sample.T[i]) / (sig * std::sqrt(sample.T[i]));
      erf1 = 0.5f + std::erf(d1 / invsqrt2) * 0.5f;
      erf2 = 0.5f + std::erf(d2 / invsqrt2) * 0.5f;

      res = sample.s0[i] * erf1 - sample.K[i] * std::exp((-1.0f) * r * sample.T[i]) * erf2;
      if (std::abs(res - sample.C(i)) > 0.1f)
      {
        std::cout << "Wrong computing " << std::endl;
        break;
      }
    }

    for (int i = N; i > N - 3; i--)
    {
      d1 = (std::log(sample.s0[i] / sample.K[i]) + (r + inv_square_sig / 2) * sample.T[i]) / (sig * std::sqrt(sample.T[i]));
      d2 = (std::log(sample.s0[i] / sample.K[i]) + (r - inv_square_sig / 2) * sample.T[i]) / (sig * std::sqrt(sample.T[i]));
      erf1 = 0.5f + std::erf(d1 / invsqrt2) * 0.5f;
      erf2 = 0.5f + std::erf(d2 / invsqrt2) * 0.5f;

      res = sample.s0[i] * erf1 - sample.K[i] * std::exp((-1.0f) * r * sample.T[i]) * erf2;
      if (std::abs(res - sample.C(i)) > 0.1f)
      {
        std::cout << "Wrong computing " << std::endl;
        break;
      }
    }
    //=================================================================================================

    std::cout << "\n";

    //Outputting time to a file (specify your path to the file)
    std::ofstream out("/common/home/durandin_v/Black-Scholes/Zameri.txt", std::ios::app);
    if (out.is_open())
    {
      out << exec_time << std::endl;
    }
    else std::cout << "File can't open!";
    out.close();

    std::cout << "Initialization time: " << init_time << "\n";
    std::cout << "Execution Time: " << exec_time << "\n";
    std::cout << "Initialization time + Execution Time: " << init_time + exec_time << std::endl;
  }

  Kokkos::finalize();
  return 0;
}

