// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/mesh_utils.hpp"

class MeshGenTest : public testing::TestWithParam<int> {
};

void checkRelativeError(const int expected, const int actual)
{
  const double error = std::abs(expected - actual) / static_cast<double>(expected);
  // This is a pretty large margin, but the disk and ball creation functions can be
  // somewhat far off
  EXPECT_LT(error, 0.5);
}

TEST_P(MeshGenTest, disk_creation)
{
  const int expected = GetParam() * 100;
  // the disk and ball meshes don't exactly hit the number
  // of elements specified, they refine to get as close as possible
  checkRelativeError(expected, serac::buildDiskMesh(expected)->GetNE());
}

TEST_P(MeshGenTest, ball_creation)
{
  const int expected = GetParam() * 400;
  // the disk and ball meshes don't exactly hit the number
  // of elements specified, they refine to get as close as possible
  checkRelativeError(expected, serac::buildBallMesh(expected)->GetNE());
}

TEST_P(MeshGenTest, rectangle_creation)
{
  int x_elts = GetParam();
  int y_elts = GetParam();

  int expected = x_elts * y_elts;
  checkRelativeError(expected, serac::buildRectangleMesh(x_elts, y_elts)->GetNE());

  // Try varying the elements for a non-square mesh
  x_elts *= 0.7;
  y_elts *= 1.19;
  expected = x_elts * y_elts;
  checkRelativeError(expected, serac::buildRectangleMesh(x_elts, y_elts)->GetNE());
}

TEST_P(MeshGenTest, cube_creation)
{
  int x_elts = GetParam();
  int y_elts = GetParam();
  int z_elts = GetParam();

  int expected = x_elts * y_elts * z_elts;
  checkRelativeError(expected, serac::buildCuboidMesh(x_elts, y_elts, z_elts)->GetNE());

  // Try varying the elements for a non-cubde mesh
  x_elts *= 0.7;
  y_elts *= 1.19;
  z_elts *= 2.46;
  expected = x_elts * y_elts * z_elts;
  checkRelativeError(expected, serac::buildCuboidMesh(x_elts, y_elts, z_elts)->GetNE());
}

INSTANTIATE_TEST_SUITE_P(BasicMeshGenTests, MeshGenTest, testing::Values(10, 20));

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when
                          // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
