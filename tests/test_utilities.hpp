// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>

#include "serac/infrastructure/input.hpp"

namespace axom {
namespace inlet {
class Inlet;
}  // namespace inlet
}  // namespace axom

namespace serac {

namespace test_utils {

void defineNonlinSolidInputFileSchema(axom::inlet::Inlet& inlet);

void runNonlinSolidTest(const std::string& input_file);

}  // end namespace test_utils

}  // end namespace serac
