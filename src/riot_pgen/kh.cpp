//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
// This file was made in part with generative AI.

#include <cmath>
#include <cstdio>


#include "riot_pgen/pgen.hpp"
#include <globals.hpp>
#include <singularity-eos/eos/eos.hpp>

namespace kh {

using parthenon::ParArray1D;
using parthenon::ParArray3D;
using namespace RiotEOS;

// ----------------------------------------------------------------------------------------
// ! \fn  void rt::ProblemGenerator
// ! Kelvin-Helmholtz Instability from Thornber & Drikakis (2008)
// ! Int. J. Numer. Meth. Fluids 2008; 56:1535-1541
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  namespace ccbulk = cell_variables::cell_averaged::bulk;
  namespace ccmat = cell_variables::cell_averaged::mat;

  // Interface surface(pin);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords; // Get the mesh coordinates

  // Collection containing fields density, velocity, pressure, magnetic field, ...
  auto &rc = pmb->meshblock_data.Get(); 

  auto &rho_left = rc->Get(ccmat::rho::name(), 0);
  if (!rho_left.IsAllocated()) {
    pmb->AllocateSparse(rho_left.label());
  }

  auto &rho_right = rc->Get(ccmat::rho::name(), 1);
  if (!rho_right.IsAllocated()) {
    pmb->AllocateSparse(rho_right.label());
  }

  // Get the EOS for each material
  auto eos_vec = pmb->packages.Get("materials")->Param<ParArray1D<EOS>>("d.d.EOS");

  // now get the pack
  auto resolved_pkgs = pmb->resolved_packages;
  static auto desc =
      riot::MakePackDescriptor<ccmat::rho, ccmat::internal_energy, ccmat::volume_fraction,
                               ccbulk::total_material_energy, ccbulk::momentum>(
          resolved_pkgs.get(), {0, 1});
  auto v = riot::GetPack(desc, rc.get());

  // Problem Parameters
  const Real rho0 = pin->GetOrAddReal("kh", "rho0", 1.0);
  const Real P0 = pin->GetOrAddReal("kh", "P0", 15);
  const Real k0 = pin->GetOrAddReal("kh", "wave_number", 2 * M_PI);
  const Real dV = pin->GetOrAddReal("kh", "mean_flow_vel", 1);

  const Real V0 = 0.1 * dV;

  pmb->par_for(
      "ProblemGenerator::kh", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        
        const Real x = coords.Xc<X1DIR>(i);
        const Real y = coords.Xc<X1DIR>(j);

        // ----------------------------------------------------------------------
        // Material
        // ----------------------------------------------------------------------
        const Real vf0 = (x < 0.0) ? 1.0 : 0.0;
        const Real vf1 = 1.0 - vf0;

        v(0, ccmat::volume_fraction(0), k, j, i) = vf0;
        v(0, ccmat::volume_fraction(1), k, j, i) = vf1;

        v(0, ccmat::rho(0), k, j, i) = rho0 * vf0;
        v(0, ccmat::rho(1), k, j, i) = rho0 * vf1;

        const Real u = energy_from_rho_P(eos_vec(0), rho0, P0);

        v(0, ccmat::internal_energy(0), k, j, i) = u;
        v(0, ccmat::internal_energy(1), k, j, i) = u;


        // ----------------------------------------------------------------------
        // Bulk
        // ----------------------------------------------------------------------
        const Real dAz_dx = - V0 * ((x > 0.0) ? 1.0 : -1.0) *
                          std::cos( k0 * y ) * std::exp( - k0 * std::abs(x) );


        const Real dAz_dy = - V0 * std::sin( k0 * y ) * std::exp( - k0 * std::abs(x) );

        const Real vx = dAz_dy;
        const Real vy = (x < 0.0) ? (-0.5 * dV - dAz_dx) : (0.5 * dV + dAz_dx);

        v(0, ccbulk::momentum(0), k, j, i) = rho0 * vx;
        v(0, ccbulk::momentum(1), k, j, i) = rho0 * vy;

        const Real ekin = 0.5 * rho0 * (vx * vx + vy * vy);

        v(0, ccbulk::total_material_energy(), k, j, i) = u + ekin;

      });

  return;
}

} // namespace kh
