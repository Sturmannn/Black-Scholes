#include <cmath>
#include <iostream>
#include <cstdlib>
#include <omp.h>

struct Option
{
<<<<<<< HEAD
  float* s0; // öåíà àêöèè â íà÷àëüíîå âðåìÿ
  float* T;  // âðåìÿ èñïîëíåíèÿ îïöèîíà â ãîäàõ
  float* K;  // ñòðàéê
  float* C;  // Ñïðàâåäëèâàÿ öåíà îïöèîíà
};

const float sig = 0.2f; // âîëàòèëüíîñòü
const float r = 0.05f; // ïðîöåíòíàÿ ñòàâêà

float start, finish; // çàìåðû âðåìåíè (çàñå÷êè)
float dt; // âðåìÿ ðàáîòû áëîêà êîäà (èçìåíåíèå âðåìåíè)
const int N = 50000000; // êîëè÷åñòâî îïöèîíîâ äëÿ ïîäñ÷¸òà

const float invsqrt2 = std::sqrt(2.0f); // èíâàðèàíòû
=======
  float* s0; // Ñ†ÐµÐ½Ð° Ð°ÐºÑ†Ð¸Ð¸ Ð² Ð½Ð°Ñ‡Ð°Ð»ÑŒÐ½Ð¾Ðµ Ð²Ñ€ÐµÐ¼Ñ
  float* T;  // Ð²Ñ€ÐµÐ¼Ñ Ð¸ÑÐ¿Ð¾Ð»Ð½ÐµÐ½Ð¸Ñ Ð¾Ð¿Ñ†Ð¸Ð¾Ð½Ð° Ð² Ð³Ð¾Ð´Ð°Ñ…
  float* K;  // ÑÑ‚Ñ€Ð°Ð¹Ðº
  float* C;  // Ð¡Ð¿Ñ€Ð°Ð²ÐµÐ´Ð»Ð¸Ð²Ð°Ñ Ñ†ÐµÐ½Ð° Ð¾Ð¿Ñ†Ð¸Ð¾Ð½Ð°
};

const float sig = 0.2f; // Ð²Ð¾Ð»Ð°Ñ‚Ð¸Ð»ÑŒÐ½Ð¾ÑÑ‚ÑŒ
const float r = 0.05f; // Ð¿Ñ€Ð¾Ñ†ÐµÐ½Ñ‚Ð½Ð°Ñ ÑÑ‚Ð°Ð²ÐºÐ°

float start, finish; // Ð·Ð°Ð¼ÐµÑ€Ñ‹ Ð²Ñ€ÐµÐ¼ÐµÐ½Ð¸ (Ð·Ð°ÑÐµÑ‡ÐºÐ¸)
float dt; // Ð²Ñ€ÐµÐ¼Ñ Ñ€Ð°Ð±Ð¾Ñ‚Ñ‹ Ð±Ð»Ð¾ÐºÐ° ÐºÐ¾Ð´Ð° (Ð¸Ð·Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ Ð²Ñ€ÐµÐ¼ÐµÐ½Ð¸)
const int N = 50000000; // ÐºÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð¾ Ð¾Ð¿Ñ†Ð¸Ð¾Ð½Ð¾Ð² Ð´Ð»Ñ Ð¿Ð¾Ð´ÑÑ‡Ñ‘Ñ‚Ð°

const float invsqrt2 = std::sqrt(2.0f); // Ð¸Ð½Ð²Ð°Ñ€Ð¸Ð°Ð½Ñ‚Ñ‹
>>>>>>> 73cb63aa4ea501dfc3f01d97286602306d5f6673
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
<<<<<<< HEAD
    sample.s0[i] = (float)rand() / (float)RAND_MAX * (150.0f - 50.0f) + 50.0f; // Ñëó÷àéíûå ÷èñëà â äèàïàçîíå
=======
    sample.s0[i] = (float)rand() / (float)RAND_MAX * (150.0f - 50.0f) + 50.0f; // Ð¡Ð»ÑƒÑ‡Ð°Ð¹Ð½Ñ‹Ðµ Ñ‡Ð¸ÑÐ»Ð° Ð² Ð´Ð¸Ð°Ð¿Ð°Ð·Ð¾Ð½Ðµ
>>>>>>> 73cb63aa4ea501dfc3f01d97286602306d5f6673
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

