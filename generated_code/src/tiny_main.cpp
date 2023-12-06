/*
 * This file was autogenerated by TinyMPC on Tue Dec  5 22:29:16 2023
 */

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/tiny_data_workspace.hpp>

using namespace Eigen;
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

#ifdef __cplusplus
extern "C" {
#endif

int main()
{
	int exitflag = 1;
	// Double check some data
	std::cout << tiny_data_solver.settings->max_iter << std::endl;
	std::cout << tiny_data_solver.cache->AmBKt.format(CleanFmt) << std::endl;
	std::cout << tiny_data_solver.work->Adyn.format(CleanFmt) << std::endl;

	exitflag = tiny_solve(&tiny_data_solver);

	if (exitflag == 0) printf("HOORAY! Solved with no error!\n");
	else printf("OOPS! Something went wrong!\n");
	return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
