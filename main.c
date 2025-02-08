#include "mpi.h"
#include "converter_mtx.h"
#include <petscksp.h>

static const char help[] = "Matrix Market file to PETSc binary converter";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec x, b; /* approx solution, RHS, exact solution */
    Mat A;       /* linear system matrix */
    KSP ksp;     /* linear solver context */
    PC pc;       /* preconditioner context */
    PetscReal norm;    /* norm of solution error */
    char matrixInputFile[PETSC_MAX_PATH_LEN];
    char vectorInputFile[PETSC_MAX_PATH_LEN];
    char solverType[PETSC_MAX_PATH_LEN] = "gmres"; /* default solver */
    char preconditionerType[PETSC_MAX_PATH_LEN] = "none"; /* default preconditioner */
    PetscReal tolerance = 1e-5; /* default tolerance */
    PetscBool flagMtx, flagVct;
    PetscInt sizen;
    MatrixInfo minfo;

    /* Initialize PETSc */
    PetscInitialize(&argc, &argv, (char *)0, help);

    /* Check how many processors are in use, one must use one node */
    MPI_Comm_size(PETSC_COMM_WORLD, &sizen);
    if (sizen > 1)
        SETERRQ(PETSC_COMM_WORLD, 1, "Uniprocessor only\n");

    /* Check for matrix and vector files */
    ierr = PetscOptionsGetString(NULL, NULL, "-matrix_file", matrixInputFile, PETSC_MAX_PATH_LEN - 1, &flagMtx);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetString(NULL, NULL, "-vector_file", vectorInputFile, PETSC_MAX_PATH_LEN - 1, &flagVct);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetString(NULL, NULL, "-solver", solverType, PETSC_MAX_PATH_LEN - 1, NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetString(NULL, NULL, "-preconditioner", preconditionerType, PETSC_MAX_PATH_LEN - 1, NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(NULL, NULL, "-tolerance", &tolerance, NULL);
    CHKERRQ(ierr);

    /* Check if files were provided */
    if (!flagMtx && !flagVct)
    {
        SETERRQ(PETSC_COMM_WORLD, 1, "No matrix file neither vector one were provided\n");
    }

    if (flagMtx)
    {
        /* Get the real matrix */
        MMTgetMatrixReal(matrixInputFile, &A, &minfo);
    }

    if (flagVct)
    {
        /* Get the real vector */
        MMTgetVectorReal(vectorInputFile, &b, &sizen);
    }

    /* Create the solver context */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp, A, A);
    CHKERRQ(ierr);

    ierr = KSPSetTolerances(ksp, tolerance, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    CHKERRQ(ierr);

    ierr = KSPSetType(ksp, solverType);
    CHKERRQ(ierr);

    ierr = KSPGetPC(ksp, &pc);
    CHKERRQ(ierr);

    ierr = PCSetType(pc, preconditionerType);
    CHKERRQ(ierr);

    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);

    /* Solve the system */
    ierr = VecDuplicate(b, &x);
    CHKERRQ(ierr);

    ierr = KSPSolve(ksp, b, x);
    CHKERRQ(ierr);

    /* Check the error */
    ierr = KSPGetResidualNorm(ksp, &norm);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Residual norm: %g\n", (double)norm);

    /*Get KSP solutions x per iteration */
    PetscInt niter;
    KSPGetIterationNumber(ksp, &niter);
    

    /* Cleanup and finalize */
    KSPDestroy(&ksp);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);

    PetscFinalize();

    return 0;
}