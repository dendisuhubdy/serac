// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <benchmark/benchmark.h>

#include "serac/infrastructure/input.hpp"

// Include this so we can check if this is a LuaJIT implementation
extern "C" {
#include "lualib.h"
}

using serac::input::CoefficientInputInfo;

constexpr static int DIM = 3;

mfem::Vector sample_vector()
{
  mfem::Vector result(DIM);
  for (int i = 0; i < DIM; i++) {
    result[i] = i * 4 + 1;
  }
  return result;
}

static void BM_DirectScalarFunction(benchmark::State& state)
{
  const CoefficientInputInfo::ScalarFunc func  = [](const mfem::Vector& vec) { return vec(1) * 2 + vec(2); };
  const auto                             input = sample_vector();
  double                                 result;
  for (auto _ : state) {
    // This code gets timed
    result = func(input);
  }
  (void)result;
}

BENCHMARK(BM_DirectScalarFunction);

static void BM_LuaScalarFunction(benchmark::State& state)
{
  // Inlet setup
  axom::sidre::DataStore datastore;
  auto                   reader = std::make_unique<axom::inlet::LuaReader>();
  reader->parseString("coef_info = { coef = function(x, y, z) return y * 2 + z end, component = 1 }");
  axom::inlet::Inlet inlet(std::move(reader), datastore.getRoot());
  auto&              coef_table = inlet.addTable("coef_info");
  CoefficientInputInfo::defineInputFileSchema(coef_table);
  auto coef_info = coef_table.get<CoefficientInputInfo>();

  const auto& func  = std::get<CoefficientInputInfo::ScalarFunc>(coef_info.func);
  const auto  input = sample_vector();
  double      result;

  for (auto _ : state) {
    // This code gets timed
    result = func(input);
  }
  (void)result;
}

BENCHMARK(BM_LuaScalarFunction);

#ifdef LUA_JITLIBNAME

static void BM_LuaJITScalarFunction(benchmark::State& state)
{
  // Inlet setup
  axom::sidre::DataStore datastore;
  auto                   reader    = std::make_unique<axom::inlet::LuaReader>();
  auto&                  sol_state = reader->solState();
  sol_state.open_libraries(sol::lib::jit, sol::lib::package, sol::lib::base);
  SLIC_ERROR_IF(sol_state["jit"].get_type() != sol::type::table, "Lua implementation is not LuaJIT!");

  reader->parseString("require('scalar_coef');");
  axom::inlet::Inlet inlet(std::move(reader), datastore.getRoot());
  auto&              coef_table = inlet.addTable("coef_info");
  CoefficientInputInfo::defineInputFileSchema(coef_table);
  auto coef_info = coef_table.get<CoefficientInputInfo>();

  const auto& func  = std::get<CoefficientInputInfo::ScalarFunc>(coef_info.func);
  const auto  input = sample_vector();
  double      result;

  for (auto _ : state) {
    // This code gets timed
    result = func(input);
  }
  (void)result;
}

BENCHMARK(BM_LuaJITScalarFunction);

#endif  // #ifdef LUA_JITLIBNAME

static void BM_DirectVectorFunction(benchmark::State& state)
{
  const CoefficientInputInfo::VecFunc func = [](const mfem::Vector& in, mfem::Vector& out) {
    out(0) = in(1) * 2;
    out(1) = in(2);
    out(2) = in(0);
  };
  const auto   input = sample_vector();
  mfem::Vector result(3);
  for (auto _ : state) {
    // This code gets timed
    func(input, result);
  }
}

BENCHMARK(BM_DirectVectorFunction);

static void BM_LuaVectorFunction(benchmark::State& state)
{
  // Inlet setup
  axom::sidre::DataStore datastore;
  auto                   reader = std::make_unique<axom::inlet::LuaReader>();
  reader->parseString("coef_info = { vec_coef = function(x, y, z) return y * 2, z, x end }");
  axom::inlet::Inlet inlet(std::move(reader), datastore.getRoot());
  auto&              coef_table = inlet.addTable("coef_info");
  CoefficientInputInfo::defineInputFileSchema(coef_table);
  auto coef_info = coef_table.get<CoefficientInputInfo>();

  const auto&  func  = std::get<CoefficientInputInfo::VecFunc>(coef_info.func);
  auto         input = sample_vector();
  mfem::Vector result(3);
  for (auto _ : state) {
    // This code gets timed
    func(input, result);
  }
}

BENCHMARK(BM_LuaVectorFunction);

#ifdef LUA_JITLIBNAME

static void BM_LuaJITVectorFunction(benchmark::State& state)
{
  // Inlet setup
  axom::sidre::DataStore datastore;
  auto                   reader    = std::make_unique<axom::inlet::LuaReader>();
  auto&                  sol_state = reader->solState();
  sol_state.open_libraries(sol::lib::jit, sol::lib::package, sol::lib::base);
  SLIC_ERROR_IF(sol_state["jit"].get_type() != sol::type::table, "Lua implementation is not LuaJIT!");

  reader->parseString("require('vec_coef');");
  axom::inlet::Inlet inlet(std::move(reader), datastore.getRoot());
  auto&              coef_table = inlet.addTable("coef_info");
  CoefficientInputInfo::defineInputFileSchema(coef_table);
  auto coef_info = coef_table.get<CoefficientInputInfo>();

  const auto&  func  = std::get<CoefficientInputInfo::VecFunc>(coef_info.func);
  auto         input = sample_vector();
  mfem::Vector result(3);
  for (auto _ : state) {
    // This code gets timed
    func(input, result);
  }
}

BENCHMARK(BM_LuaJITVectorFunction);

#endif  // #ifdef LUA_JITLIBNAME

BENCHMARK_MAIN();
