#ifndef GUARD_bd_h
#define GUARD_bd_h

#include <random>
#include "info.h"
#include "tree.h"

#ifdef MPIBART
bool bd(tree& x, xinfo& xi, pinfo& pi, std::default_random_engine& gen, size_t numslaves);
#else
bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, std::default_random_engine& gen);
#endif

#endif
