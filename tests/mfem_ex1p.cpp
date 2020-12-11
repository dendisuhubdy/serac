//                       MFEM Example 1 - Parallel Version
//                              AmgX Modification
//
// Compile with: make ex1p
//
// AmgX sample runs:
//               mpirun -np 4 ex1p
//               mpirun -np 4 ex1p -d cuda
//               mpirun -np 10 ex1p --amgx-file amg_pcg.json --amgx-mpi-teams
//               mpirun -np 4 ex1p --amgx-file amg_pcg.json
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "numerics/mesh_utils.hpp"
#include "physics/thermal_conduction.hpp"
#include "numerics/expr_template_ops.hpp"
#include "serac_config.hpp"
#include <mfem/linalg/dtensor.hpp>


using namespace std;
using namespace mfem;

#ifndef MFEM_USE_AMGX
#error This example requires that MFEM is built with MFEM_USE_AMGX=YES
#endif

/* Start of dark arts */
template < typename From, auto V, typename Result, auto V2, typename Result2> 
struct forbidden
{
  friend Result _get_pa_data(From& from)
  {
    return from.*V;
  }

  friend Result2 _get_intrule(From & from)
  {
    return from.*V2;
  }

};

mfem::Vector _get_pa_data(mfem::DiffusionIntegrator&);
const mfem::IntegrationRule * _get_intrule(mfem::DiffusionIntegrator&);

template struct forbidden<mfem::DiffusionIntegrator, &mfem::DiffusionIntegrator::pa_data, mfem::Vector , &mfem::DiffusionIntegrator::IntRule, const mfem::IntegrationRule *>;

/* End of dark arts */

// Build a sparse matrix from the diffusion integrator
mfem::SparseMatrix PA_Assemble(mfem::ParFiniteElementSpace & pfes, mfem::DiffusionIntegrator & integrator)
{
  auto intrule = _get_intrule(integrator);
  const mfem::FiniteElement &el = *pfes.GetFE(0); // only works if all elements have the same integration rule
  intrule = intrule ? intrule : &mfem::DiffusionIntegrator::GetRule(el,el);
  int nq = intrule->GetNPoints();
  
  // We assume all the elements in the finite element space are the same. This is the same assumption PA makes atm.
  auto FE_0 = pfes.GetFE(0);
  int nvdofs = FE_0->GetDim() * FE_0->GetDof();

  mfem::SparseMatrix assemble_mat(pfes.GetVSize(), pfes.GetVSize());
  auto pa_data = _get_pa_data(integrator);
  std::cout << pa_data.Size() << " " << nq << " " << nvdofs << " " << pfes.GetNE() << std::endl;

  auto P = mfem::Reshape(pa_data.Read(), nq, 6, pfes.GetNE());

  /// mfem's hard coded numbering
  int inv_d[] = { 0, 1, 2,
		  1, 3, 4,
		  2, 4, 5 };
  
  for (int e = 0; e < pfes.GetNE(); e++) {
    mfem::DenseMatrix el_mat (nvdofs, nvdofs);
    el_mat = 0.;
    auto FE = pfes.GetFE(e);
    // loop over dofs..
    // JW: This is the way mfem does it for this ex1p's instance. We should come up with a generalized mapping to do this appropriately.
    for (int q = 0; q < nq; q++) {

      const IntegrationPoint &ip = intrule->IntPoint(q);
      
      DenseMatrix dshape(FE->GetDof(), FE->GetDim());
      FE->CalcDShape(ip, dshape);
      
      for (int iv = 0; iv < nvdofs; iv++) {
      
	for (int jv = 0; jv < nvdofs; jv++) {

	  for (int di = 0; di < 3; di++) {
	    for (int dj = 0; dj < 3 ; dj++) {
	      el_mat(iv, jv) +=
		dshape(iv, di)  
		* P(q, inv_d[dj + 3 * di], e)
		* dshape(jv, dj);
	    }
	  }
	}
      }
    }

    
    Array<int> elem_vdofs;
    pfes.GetElementVDofs(e, elem_vdofs);
    assemble_mat.AddSubMatrix(elem_vdofs, elem_vdofs, el_mat);
  }

  assemble_mat.Finalize();
  
  return assemble_mat;
}

std::unique_ptr<mfem::SparseMatrix> SparseMat_EliminiateVDOFS(mfem::Array<int> & true_dofs_list, mfem::ParFiniteElementSpace * pfes, std::unique_ptr<mfem::SparseMatrix> A)
{
  const SparseMatrix *P = pfes->GetConformingProlongation();
  if (P) {
    SparseMatrix *R  = mfem::Transpose(*P);
    SparseMatrix *RA = mfem::Mult(*R, *A);
    delete R;
    A.reset(mfem::Mult(*RA, *P));
    delete RA;
  } 
  // Eliminate the degrees of freedom
  for (auto r : true_dofs_list)
    {
      A->EliminateRowCol(r);
    }
  return A;
}


int main(int argc, char *argv[])
{
  // 1. Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Parse command-line options.
  const char *mesh_file = "../../data/meshes/onehex.mesh";
  int order = 1;
  bool static_cond = false;
  bool pa = false;
  const char *device_config = "cpu";
  bool visualization = true;
  bool amgx_lib = true;
  bool amgx_mpi_teams = false;
  const char* amgx_json_file = ""; // JSON file for AmgX
  int ndevices = 1;
  bool dark_arts = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
		 "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
		 "Finite element order (polynomial degree) or -1 for"
		 " isoparametric space.");
  args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
		 "--no-static-condensation", "Enable static condensation.");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
		 "--no-partial-assembly", "Enable Partial Assembly.");
  args.AddOption(&dark_arts, "-magic", "--magic", "-no-magic",
		 "--no-magic", "Enable Magic (PA) -> CSR.");   
  args.AddOption(&amgx_lib, "-amgx", "--amgx-lib", "-no-amgx",
		 "--no-amgx-lib", "Use AmgX in example.");
  args.AddOption(&amgx_json_file, "--amgx-file", "--amgx-file",
		 "AMGX solver config file (overrides --amgx-solver, --amgx-verbose)");
  args.AddOption(&amgx_mpi_teams, "--amgx-mpi-teams", "--amgx-mpi-teams",
		 "--amgx-mpi-gpu-exclusive", "--amgx-mpi-gpu-exclusive",
		 "Create MPI teams when using AmgX to load balance between ranks and GPUs.");
  args.AddOption(&device_config, "-d", "--device",
		 "Device configuration string, see Device::Configure().");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		 "--no-visualization",
		 "Enable or disable GLVis visualization.");
  args.AddOption(&ndevices, "-nd","--gpus-per-node-in-teams-mode",
		 "Number of GPU devices per node (Only used if amgx_mpi_teams is true).");

  args.Parse();
  if (!args.Good())
    {
      if (myid == 0)
	{
	  args.PrintUsage(cout);
	}
      MPI_Finalize();
      return 1;
    }
  if (myid == 0)
    {
      args.PrintOptions(cout);
    }

  // 3. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(device_config);
  if (myid == 0) { device.Print(); }

  // 4. Read the (serial) mesh from the given mesh file on all processors.  We
  //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
  //    and volume meshes with the same code.
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  // 5. Refine the serial mesh on all processors to increase the resolution. In
  //    this example we do 'ref_levels' of uniform refinement. We choose
  //    'ref_levels' to be the largest number that gives a final mesh with no
  //    more than 10,000 elements.
  {
    int ref_levels = 1;
    //         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
    for (int l = 0; l < ref_levels; l++)
      {
	mesh.UniformRefinement();
      }
  }

  // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  {
    int par_ref_levels = 0;
    for (int l = 0; l < par_ref_levels; l++)
      {
	pmesh.UniformRefinement();
      }
  }

  // 7. Define a parallel finite element space on the parallel mesh. Here we
  //    use continuous Lagrange finite elements of the specified order. If
  //    order < 1, we instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  bool delete_fec;
  if (order > 0)
    {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
    }
  else if (pmesh.GetNodes())
    {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
	{
	  cout << "Using isoparametric FEs: " << fec->Name() << endl;
	}
    }
  else
    {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
    }
  ParFiniteElementSpace fespace(&pmesh, fec);
  HYPRE_Int size = fespace.GlobalTrueVSize();
  if (myid == 0)
    {
      cout << "Number of finite element unknowns: " << size << endl;
    }

  // 8. Determine the list of true (i.e. parallel conforming) essential
  //    boundary dofs. In this example, the boundary conditions are defined
  //    by marking all the boundary attributes from the mesh as essential
  //    (Dirichlet) and converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (pmesh.bdr_attributes.Size())
    {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

  // 9. Set up the parallel linear form b(.) which corresponds to the
  //    right-hand side of the FEM linear system, which in this case is
  //    (1,phi_i) where phi_i are the basis functions in fespace.
  ParLinearForm b(&fespace);
  ConstantCoefficient one(1.0);
  b.AddDomainIntegrator(new DomainLFIntegrator(one));
  b.Assemble();

  // 10. Define the solution vector x as a parallel finite element grid function
  //     corresponding to fespace. Initialize x with initial guess of zero,
  //     which satisfies the boundary conditions.
  ParGridFunction x(&fespace);
  x = 0.0;

  // 11. Set up the parallel bilinear form a(.,.) on the finite element space
  //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //     domain integrator.
  ParBilinearForm a(&fespace);
  if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
  a.AddDomainIntegrator(new DiffusionIntegrator(one));

  // 12. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  if (static_cond) { a.EnableStaticCondensation(); }
  a.Assemble();

  OperatorPtr A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  // 13. Solve the linear system A X = B.
  //     * With full assembly, use the BoomerAMG preconditioner from hypre.
  //     * If AmgX is available solve using amg preconditioner.
  //     * With partial assembly, use Jacobi smoothing, for now.
  Solver *prec = NULL;
  if (pa)
    {
      if (!dark_arts) {

	if (UsesTensorBasis(fespace))
	  {
	    prec = new OperatorJacobiSmoother(a, ess_tdof_list);
	  }

	CGSolver cg(MPI_COMM_WORLD);
	cg.SetRelTol(1e-12);
	cg.SetMaxIter(2000);
	cg.SetPrintLevel(1);
	if (prec) { cg.SetPreconditioner(*prec); }
	cg.SetOperator(*A);
	cg.Mult(B, X);
	delete prec;

      } else { // magic
       
	auto integrators = a.GetDBFI();
	auto sp_mat = std::make_unique<mfem::SparseMatrix>();
	auto sp_mat2 = std::make_unique<mfem::SparseMatrix>();
	*sp_mat = PA_Assemble(fespace, *static_cast<mfem::DiffusionIntegrator *>((*integrators)[0]));
	*sp_mat2 = *sp_mat;
	sp_mat2 = SparseMat_EliminiateVDOFS(ess_tdof_list, &fespace, std::move(sp_mat2));
	//SparseMatrix A_sp(*sp_mat);
	Operator *sp_oper;
	Vector B_sp, X_sp;
	sp_mat->FormLinearSystem(ess_tdof_list, x, b, sp_oper, X_sp, B_sp);
     
	mfem::OperatorPtr A_sp;
	A_sp.Reset(sp_oper);

	bool amgx_verbose = false;
	prec = new AmgXSolver(MPI_COMM_WORLD, AmgXSolver::PRECONDITIONER,
			      amgx_verbose);

	// if (UsesTensorBasis(fespace))
	//   {
	//     prec = new OperatorJacobiSmoother(a, ess_tdof_list);
	//   }

	
	std::cout << "starting the solver here!" << std::endl;
	
	CGSolver cg(MPI_COMM_WORLD);
	cg.SetRelTol(1e-12);
	cg.SetMaxIter(2000);
	cg.SetPrintLevel(1);
	if (prec) { cg.SetPreconditioner(*prec); }
	//	cg.SetOperator(*sp_oper);
	cg.SetOperator(*sp_mat2);
	cg.Mult(B, X);
	delete prec;
      }
   }
   else if (amgx_lib && strcmp(amgx_json_file,"") == 0)
   {
      MFEM_VERIFY(!amgx_mpi_teams,
                  "Please add JSON file to try AmgX with MPI teams mode");

      bool amgx_verbose = false;
      prec = new AmgXSolver(MPI_COMM_WORLD, AmgXSolver::PRECONDITIONER,
                            amgx_verbose);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(1);
      if (prec) { cg.SetPreconditioner(*prec); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;

   }
   else if (amgx_lib && strcmp(amgx_json_file,"") != 0)
   {
      AmgXSolver amgx;
      amgx.ReadParameters(amgx_json_file, AmgXSolver::EXTERNAL);

      if (amgx_mpi_teams)
      {
         // Forms MPI teams to load balance between MPI ranks and GPUs
         amgx.InitMPITeams(MPI_COMM_WORLD, ndevices);
      }
      else
      {
         // Assumes each MPI rank is paired with a GPU
         amgx.InitExclusiveGPU(MPI_COMM_WORLD);
      }

      amgx.SetOperator(*A.As<HypreParMatrix>());
      amgx.Mult(B, X);

      // Release MPI communicators and resources created by AmgX
      amgx.Finalize();
   }
   else
   {
      prec = new HypreBoomerAMG;

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(1);
      if (prec) { cg.SetPreconditioner(*prec); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);
   std::cout << "X Norm: " << x.Norml2() << std::endl;
   
   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }
   MPI_Finalize();

   return 0;
}
