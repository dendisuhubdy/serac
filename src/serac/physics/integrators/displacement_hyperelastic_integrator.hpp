// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file inc_hyperelastic_integrator.hpp
 *
 * @brief The MFEM integrators for the incremental hyperelastic formulation
 */

#ifndef DISPLACEMENT_INTEGRATOR_HPP
#define DISPLACEMENT_INTEGRATOR_HPP

#include "serac/physics/materials/hyperelastic_material.hpp"

#include "mfem.hpp"

namespace serac {

/**
 * @brief Incremental hyperelastic integrator for any given HyperelasticModel.
 *
 * Represents @f$ \int W(Jpt) dx @f$ over a target zone, where W is the
 * @a model's strain energy density function, and Jpt is the Jacobian of the
 * target->physical coordinates transformation. The target configuration is
 * given by the current mesh at the time of the evaluation of the integrator.
 */
class DisplacementHyperelasticIntegrator : public mfem::NonlinearFormIntegrator {
public:
  /**
   * @brief The constructor for the incremental hyperelastic integrator
   *
   * @param[in] m  HyperelasticModel that will be integrated.
   */
  explicit DisplacementHyperelasticIntegrator(serac::HyperelasticMaterial& m, const int dim, bool geom_nonlin = true)
      : material_(m), geom_nonlin_(geom_nonlin)
  {
    getShearTerms(dim, shear_terms_);
    eye_.SetSize(dim);
    eye_ = 0.0;
    for (int i = 0; i < dim; ++i) {
      eye_(i, i) = 1.0;
    }
  }

  /**
   * @brief Computes the integral of W(Jacobian(Trt)) over a target zone
   * @param[in] el     Type of FiniteElement.
   * @param[in] Ttr    Represents ref->target coordinates transformation.
   * @param[in] elfun  Physical coordinates of the zone.
   */
  virtual double GetElementEnergy(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
                                  const mfem::Vector& elfun);

  /**
   * @brief The residual evaluation for the nonlinear incremental integrator
   *
   * @param[in] el The finite element to integrate
   * @param[in] Ttr The element transformation operators
   * @param[in] elfun The state vector to evaluate the residual
   * @param[out] elvect The output residual
   */
  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
                                     const mfem::Vector& elfun, mfem::Vector& elvect);

  /**
   * @brief Assemble the local gradient
   *
   * @param[in] el The finite element to integrate
   * @param[in] Ttr The element transformation operators
   * @param[in] elfun The state vector to evaluate the gradient
   * @param[out] elmat The output local gradient
   */
  virtual void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
                                   const mfem::Vector& elfun, mfem::DenseMatrix& elmat);

private:
  void CalcDeformationGradient(const mfem::FiniteElement& el, const mfem::IntegrationPoint& ip,
                               mfem::ElementTransformation& Ttr);

  void CalcBMatrix();

  /**
   * @brief The associated hyperelastic model
   */
  serac::HyperelasticMaterial& material_;

  /**
   * Jrt: the Jacobian of the target-to-reference-element transformation.
   * Jpr: the Jacobian of the reference-to-physical-element transformation.
   * F: the Jacobian of the target-to-physical-element transformation (deformation gradient) (dx_i/DX_j).
   * P: represents dW_d(Jtp) (dim x dim).
   * DSh: gradients of reference shape functions (dof x dim).
   * DS: gradients of the shape functions in the target configuration (DN_i, DX_j)
   * B0_T: B matrix in Voigt notation
   * configuration (dof x dim).
   * PMatI: coordinates of the deformed configuration (dof x dim).
   * PMatO: reshaped view into the local element contribution to the operator
   * output - the result of AssembleElementVector() (dof x dim).
   */
  mfem::DenseMatrix DSh_, DS_, Jrt_, Jpr_, F_, C_, P_, T_, PMatI_, PMatO_, temp_, K_, B_0T, H_, HT_, eye_, S_mat_;

  std::vector<std::pair<int, int>> shear_terms_;

  std::vector<mfem::DenseMatrix> B_0_;

  mfem::Vector S_, force_, vec_temp_;

  /**
   * @brief The geometric nonlinearity flag
   */
  bool geom_nonlin_;
};

}  // namespace serac

#endif