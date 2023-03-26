
#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>

int main(int argc, char* argv[]) {
  Kokkos::initialize(
    Kokkos::InitializationSettings()
    .set_disable_warnings(false)
    .set_num_threads(4));
  //Kokkos::initialize(argc, argv);
  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << "[<kokkos_options>] <size>" << std::endl;
    Kokkos::finalize();
    exit(1);
  }

  const int n = std::stoi(argv[1], nullptr, 10);

  std::cout << "Number of even integers from 0 to " << n - 1 << std::endl;

  Kokkos::View<float*> v("V", n), seqRes("SeqRes", n), parRes("ParRes", n);

  for (int i = 0; i < n; i++)
    v(i) = 0.0;

  float expRes = 0.0f;

  // sequential

  Kokkos::Timer timer;
  timer.reset();

  for (int i = 0; i < n; i++) {
    seqRes(i) = Kokkos::log(Kokkos::exp(Kokkos::log(Kokkos::exp(Kokkos::log(Kokkos::exp(v(i)))))));
  }

  double count_time = timer.seconds();
  std::cout << "  Sequential: " << count_time << std::endl;

  for (int i = 0; i < n; i++)
    if (std::abs(seqRes(i) - expRes) > 1e-8) {
      std::cout << "Error in sequential " << i << ": " << seqRes(i) << " instead of " << expRes << std::endl;
      Kokkos::finalize();
      exit(1);
    }

  timer.reset();

  // parallel

  Kokkos::parallel_for(
    n, KOKKOS_LAMBDA(const int i) {
    parRes(i) = Kokkos::log(Kokkos::exp(Kokkos::log(Kokkos::exp(Kokkos::log(Kokkos::exp(v(i)))))));
  });

  count_time = timer.seconds();
  std::cout << "  Parallel: " << count_time << std::endl;

  for (int i = 0; i < n; i++)
    if (std::abs(parRes(i) - expRes) > 1e-8) {
      std::cout << "Error in parallel " << i << ": " << parRes(i) << " instead of " << expRes << std::endl;
      Kokkos::finalize();
      exit(1);
    }

  Kokkos::finalize();
}




////@HEADER
//// ************************************************************************
////
////                        Kokkos v. 4.0
////       Copyright (2022) National Technology & Engineering
////               Solutions of Sandia, LLC (NTESS).
////
//// Under the terms of Contract DE-NA0003525 with NTESS,
//// the U.S. Government retains certain rights in this software.
////
//// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
//// See https://kokkos.org/LICENSE for license information.
//// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
////@HEADER
//
//#include <Kokkos_Core.hpp>
//#include <cstdio>
////#include <iostream>
////#include <Kokkos_Vector.hpp>
//
//int main(int argc, char* argv[]) {
//  Kokkos::initialize(argc, argv);
//  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);
//
//
//  //Kokkos::vector<int> v;
//  //v.push_back(55);
//  //std::cout << v.size() << "\n";
//
//
//
//  if (argc < 2) {
//    fprintf(stderr, "Usage: %s [<kokkos_options>] <size>\n", argv[0]);
//    Kokkos::finalize();
//    exit(1);
//  }
//
//  const long n = strtol(argv[1], nullptr, 10);
//
//  printf("Number of even integers from 0 to %ld\n", n - 1);
//
//  Kokkos::Timer timer;
//  timer.reset();
//
//  // Compute the number of even integers from 0 to n-1, in parallel.
//  long count = 0;
//  Kokkos::parallel_reduce(
//    n, KOKKOS_LAMBDA(const long i, long& lcount) { lcount += (i % 2) == 0; },
//    count);
//
//  double count_time = timer.seconds();
//  printf("  Parallel: %ld    %10.6f\n", count, count_time);
//
//  timer.reset();
//
//  // Compare to a sequential loop.
//  long seq_count = 0;
//  for (long i = 0; i < n; ++i) {
//    seq_count += (i % 2) == 0;
//  }
//
//  count_time = timer.seconds();
//  printf("Sequential: %ld    %10.6f\n", seq_count, count_time);
//
//  Kokkos::finalize();
//
//  return (count == seq_count) ? 0 : -1;
//}
