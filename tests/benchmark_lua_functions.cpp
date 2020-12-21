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
  const CoefficientInputInfo::ScalarFunc func = [](const mfem::Vector& vec) {
    auto x  = vec(0);
    auto y  = vec(1);
    auto z  = vec(2);
    auto x2 = x;
    auto y2 = y;
    auto z2 = z;
    for (int i = 0; i < 100; i++) {
      x2 += i;
      y2 += i * 2;
      z2 += i * 3;
    }

    auto comp0 = (y * z2) - (z * y2);
    auto comp1 = (x * z2) - (z * x2);
    auto comp2 = (x * y2) - (y * z2);
    return comp0 + comp1 + comp2;
  };
  const auto input = sample_vector();
  double     result;
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
  reader->parseString(
      "coef_info = {\n"
      "  coef = function(x, y, z)\n"
      "    local x2 = x\n"
      "    local y2 = y\n"
      "    local z2 = z\n"
      "    for i = 1, 100 do\n"
      "      x2 = x2 + i\n"
      "      y2 = y2 + (2 * i)\n"
      "      z2 = z2 + (3 * i)\n"
      "    end\n"
      "    -- cross product\n"
      "    local comp0 = (y * z2) - (z * y2)\n"
      "    local comp1 = (x * z2) - (z * x2)\n"
      "    local comp2 = (x * y2) - (y * z2)\n"
      "    return comp0 + comp1 + comp2\n"
      "  end,\n"
      "  component = 1,"
      "}\n");
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
    auto x  = in(0);
    auto y  = in(1);
    auto z  = in(2);
    auto x2 = x;
    auto y2 = y;
    auto z2 = z;
    for (int i = 0; i < 100; i++) {
      x2 += i;
      y2 += i * 2;
      z2 += i * 3;
    }

    out(0) = (y * z2) - (z * y2);
    out(1) = (x * z2) - (z * x2);
    out(2) = (x * y2) - (y * z2);
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
  reader->parseString(
      "coef_info = {\n"
      "  vec_coef = function(x, y, z)\n"
      "    local x2 = x\n"
      "    local y2 = y\n"
      "    local z2 = z\n"
      "    for i = 1, 100 do\n"
      "      x2 = x2 + i\n"
      "      y2 = y2 + (2 * i)\n"
      "      z2 = z2 + (3 * i)\n"
      "    end\n"
      "    -- cross product\n"
      "    local comp0 = (y * z2) - (z * y2)\n"
      "    local comp1 = (x * z2) - (z * x2)\n"
      "    local comp2 = (x * y2) - (y * z2)\n"
      "    return comp0, comp1, comp2\n"
      "  end\n"
      "}\n");
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
