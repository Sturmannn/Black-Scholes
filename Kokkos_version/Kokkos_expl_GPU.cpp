#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <fstream>

const float sig = 0.2f; // волатильность
const float r = 0.05f; // процентная ставка
const int N = 50000000; // количество опционов для подсчёта

const float inv_square_sig = sig * sig;
//Kokkos::View<float*, Kokkos::SharedSpace> e("e", N);
//Kokkos::View<const float, Kokkos::CudaHostPinnedSpace> x("x");
const float invsqrt2 = 1.414213f;


struct Option
{
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> s0; // цена акции в начальное время
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> T;  // время исполнения опциона в годах
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> K;  // страйк
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> C;  // Справедливая цена опциона

  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror host_s0;
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror host_T;
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror host_K;
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror host_C;
};

void Set_Count_Of_Options(Option& opt)
{
  Kokkos::resize(Kokkos::WithoutInitializing, opt.C, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.K, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.T, N);
  Kokkos::resize(Kokkos::WithoutInitializing, opt.s0, N);

  opt.host_C = Kokkos::create_mirror_view(opt.C);
  opt.host_K = Kokkos::create_mirror_view(opt.K);
  opt.host_T = Kokkos::create_mirror_view(opt.T);
  opt.host_s0 = Kokkos::create_mirror_view(opt.s0);
}

void Deep_Copy_Device_To_Host(Option& opt)
{
  Kokkos::deep_copy(opt.host_C, opt.C);
  Kokkos::deep_copy(opt.host_K, opt.K);
  Kokkos::deep_copy(opt.host_T, opt.T);
  Kokkos::deep_copy(opt.host_s0, opt.s0);
}

void Deep_Copy_Host_To_Device(Option& opt)
{
  Kokkos::deep_copy(opt.C, opt.host_C);
  Kokkos::deep_copy(opt.K, opt.host_K);
  Kokkos::deep_copy(opt.T, opt.host_T);
  Kokkos::deep_copy(opt.s0, opt.host_s0);
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
      Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace> generator = random_pool.get_state();

      sample.K(i) = (float)generator.rand() / (float)generator.MAX_RAND * (250.0f - 50.0f) + 50.0f;
      sample.s0(i) = (float)generator.rand() / (float)generator.MAX_RAND * (150.0f - 50.0f) + 50.0f; // Случайные числа в диапазоне
      sample.T(i) = (float)generator.rand() / (float)generator.MAX_RAND * (5.0f - 1.0f) + 1.0f;
      random_pool.free_state(generator);
    });

    Kokkos::fence();

    std::cout << "Device init execution time: " << (float)timer.seconds() << "\n";

    timer.reset();
    Deep_Copy_Device_To_Host(sample); // Предполагается, что на этом этапе нам понадобятся эти данные на хосте (для реальных задач)
    std::cout << "Deep copy to host after device-init: " << (float)timer.seconds() << "\n";

    Kokkos::fence();

    timer.reset();
    GetOptionPrices(sample);
    Kokkos::fence();
    float exec_time = (float)timer.seconds();
    std::cout << "Device execution time: " << exec_time << "\n";

    timer.reset();
    Deep_Copy_Device_To_Host(sample); // Предполагается, что на этом этапе нам понадобятся эти данные на хосте (для реальных задач)
    std::cout << "Deep copy to host after device-exec: " << (float)timer.seconds() << "\n";

    timer.reset();
    Deep_Copy_Host_To_Device(sample); // Для сравнения времени передачи с устройства на хост и наоборот
    std::cout << "Deep copy to device after device-host: " << (float)timer.seconds() << "\n";


    for (int i = 0; i < 5; i++)
    {
      std::cout << "C =  " << sample.C(i) << std::endl;
      std::cout << "K =  " << sample.K(i) << std::endl;
      std::cout << "s0 =  " << sample.s0(i) << std::endl;
      std::cout << "T =  " << sample.T(i) << std::endl;
      std::cout << std::endl;
    }

    std::cout << "\n";

    if (Kokkos::abs(sample.C(0) - 0.673959f) > 0.1f) std::cout << "WRONG 0!!!\n";

    std::ofstream out("/common/home/durandin_v/Black-Scholes/Zameri.txt", std::ios::app);
    if (out.is_open())
    {
      out << exec_time << std::endl;
    }
    else std::cout << "File can't open!";
    out.close();
  }

  Kokkos::finalize();
  return 0;
}

