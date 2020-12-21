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
#define luaJIT_BC_scalar_coef_SIZE 86
  static const char luaJIT_BC_scalar_coef[] = {
      27,  76,  74,  2, 2,  20, 0,  3,  4,   0,   0,   1,   3,   24,  3,   0,   1,   32,  3,   2,   3,  76,
      3,   2,   0,   4, 58, 3,  0,  2,  0,   4,   0,   5,   53,  0,   1,   0,   51,  1,   0,   0,   61, 1,
      2,   0,   55,  0, 3,  0,  75, 0,  1,   0,   14,  99,  111, 101, 102, 95,  105, 110, 102, 111, 9,  99,
      111, 101, 102, 1, 0,  1,  14, 99, 111, 109, 112, 111, 110, 101, 110, 116, 3,   1,   0,   0};
  // Inlet setup
  axom::sidre::DataStore datastore;
  auto                   reader    = std::make_unique<axom::inlet::LuaReader>();
  auto&                  sol_state = reader->solState();
  sol_state.open_libraries(sol::lib::jit, sol::lib::package, sol::lib::base);
  SLIC_ERROR_IF(sol_state["jit"].get_type() != sol::type::table, "Lua implementation is not LuaJIT!");
  // This does not work through sol so we need to get at the luaState directly
  // sol_state["package"]["preload"]["scalar_coef"] = sol_state.load_buffer(luaJIT_BC_scalar_coef,
  // luaJIT_BC_scalar_coef_SIZE);
  auto lua_state = sol_state.lua_state();
  lua_getglobal(lua_state, "package");
  lua_getfield(lua_state, -1, "preload");
  bool error = luaL_loadbuffer(lua_state, luaJIT_BC_scalar_coef, luaJIT_BC_scalar_coef_SIZE, NULL);
  SLIC_ERROR_IF(error, "Failed to load LuaJIT bytecode");
  lua_setfield(lua_state, -2, "scalar_coef");

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
#define luaJIT_BC_vec_coef_SIZE 82
  static const char luaJIT_BC_vec_coef[] = {
      27, 76,  74,  2,   2,   24, 0,   3,   6,  0,  0,  1,   4,   24,  3, 0, 1,  18, 4,   2,   0,
      18, 5,   0,   0,   74,  3,  4,   0,   4,  50, 3,  0,   2,   0,   4, 0, 5,  53, 0,   1,   0,
      51, 1,   0,   0,   61,  1,  2,   0,   55, 0,  3,  0,   75,  0,   1, 0, 14, 99, 111, 101, 102,
      95, 105, 110, 102, 111, 13, 118, 101, 99, 95, 99, 111, 101, 102, 1, 0, 0,  0,  0};
  // Inlet setup
  axom::sidre::DataStore datastore;
  auto                   reader    = std::make_unique<axom::inlet::LuaReader>();
  auto&                  sol_state = reader->solState();
  sol_state.open_libraries(sol::lib::jit, sol::lib::package, sol::lib::base);
  SLIC_ERROR_IF(sol_state["jit"].get_type() != sol::type::table, "Lua implementation is not LuaJIT!");
  // This does not work through sol so we need to get at the luaState directly
  // sol_state["package"]["preload"]["scalar_coef"] = sol_state.load_buffer(luaJIT_BC_vec_coef,
  // luaJIT_BC_vec_coef_SIZE);
  auto lua_state = sol_state.lua_state();
  lua_getglobal(lua_state, "package");
  lua_getfield(lua_state, -1, "preload");
  bool error = luaL_loadbuffer(lua_state, luaJIT_BC_vec_coef, luaJIT_BC_vec_coef_SIZE, NULL);
  SLIC_ERROR_IF(error, "Failed to load LuaJIT bytecode");
  lua_setfield(lua_state, -2, "vec_coef");

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
