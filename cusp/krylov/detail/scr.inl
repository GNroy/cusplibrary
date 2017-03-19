/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{
	
struct _method {};

struct Restarts : public _method {};
struct Limited_Orthogonalization : public _method {};



template <typename Method,
		 class LinearOperator,
         class Vector>
void scr(LinearOperator& A,
        Vector& x,
        Vector& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::default_monitor<ValueType> monitor(b);

    cusp::krylov::scr(A, x, b, monitor);
}

template <typename Method,
		 class LinearOperator,
         class Vector,
         class Monitor>
void scr(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    cusp::krylov::scr(A, x, b, monitor, M);
}

template <typename Method,
		 class LinearOperator,
         class Vector,
         class Monitor,
         class Preconditioner>
void scr(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M)
{
	const unsigned int recompute_r = 8;
	
	cusp::krylov::scr(A, x, b, monitor, M, recompute_r);
}

template <typename Method,
		 class LinearOperator,
         class Vector,
         class Monitor,
         class Preconditioner,
		 typename IndexType>
void scr(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M,
		IndexType recompute_r)	     // interval to update r
{
    CUSP_PROFILE_SCOPED();

    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> Ap(N);
    cusp::array1d<ValueType,MemorySpace> z(N);
    cusp::array1d<ValueType,MemorySpace> r(N);
	cusp::array1d<ValueType,MemorySpace> p(N);
	
	typedef cusp::array1d<ValueType,MemorySpace> array1dSCR;
	array1dSCR pN[recompute_r];
	array1dSCR ApN[recompute_r];
	for (int i=0; i<recompute_r; ++i)
	{
		ApN[i] = cusp::array1d<ValueType, MemorySpace> (N);
		pN[i] = cusp::array1d<ValueType, MemorySpace> (N);
	}
	
	cusp::array1d<ValueType,MemorySpace> dotc_Ap(recompute_r);
	cusp::array1d<ValueType,MemorySpace> beta(N);
    cusp::array1d<ValueType,MemorySpace> Ax(N);

    // Ax <- A*x
    cusp::multiply(A, x, Ax);

    // r <- b - Ax
    blas::axpby(b, Ax, r, ValueType(1), ValueType(-1));

    // z <- M*r
    cusp::multiply(M, r, z);

    // p <- z
    blas::copy(z, p);

    // Ap <- A*p
    cusp::multiply(A, p, Ap);
	
	// pN[0] <- p
	blas::copy(p, pN[monitor.iteration_count()]);
	
	// ApN[0] <- Ap
	blas::copy(Ap, ApN[monitor.iteration_count()]);
	
	dotc_Ap[monitor.iteration_count()] = blas::dotc(Ap, Ap);
		
	// alpha <- <r,Ap>/<Ap,Ap>
	ValueType alpha =  blas::dotc(r, Ap) / dotc_Ap[monitor.iteration_count()];
	
	if( thrust::detail::is_same<Method, Restarts>::value )
	{
		while (!monitor.finished(r))
		{
			size_t local_iter = (monitor.iteration_count() + 1) % recompute_r;
			
			if (local_iter)
			{
				// x <- x + alpha * p
				blas::axpy(p, x, alpha);
				
				// r <- r - alpha * Ap
				blas::axpy(Ap, r, -alpha);
			}
			else
			{
				// Ax <- A*x
				cusp::multiply(A, x, Ax);

				// r <- b - A*x
				blas::axpby(b, Ax, r, ValueType(1), ValueType(-1));
			}
	
			// z <- M*r
			cusp::multiply(M, r, z);
			
			// p_0 <- z
			blas::copy(z, p);
			
			// Ap <- A*p
			cusp::multiply(A, p, Ap);
			
			for (int i=0; i<local_iter; ++i)
			{
				ValueType beta = - blas::dotc(Ap, ApN[i]) / dotc_Ap[i];
				
				blas::axpy(pN[i], p, beta);
				
				blas::axpy(ApN[i], Ap, beta);
			}
			
			blas::copy(p, pN[local_iter]);
			
			blas::copy(Ap, ApN[local_iter]);
			
			dotc_Ap[local_iter] = blas::dotc(Ap, Ap);
			
			// alpha <- <r,Ap>/<Ap,Ap>
			alpha =  blas::dotc(r, Ap) / dotc_Ap[local_iter];
			
			++monitor;
		}
	}
	else if( thrust::detail::is_same<Method, Limited_Orthogonalization>::value )
	{
		int i;
		bool flag=false;
		
		while (!monitor.finished(r))
		{
			if (!flag || i == recompute_r) i=0;

			// x <- x + alpha * p
			blas::axpy(p, x, alpha);

			// r <- r - alpha * Ap
			blas::axpy(Ap, r, -alpha);
			
			// z <- M*r
			cusp::multiply(M, r, z);
			
			// p_0 <- z
			blas::copy(z, p);
			
			// Ap <- A*p
			cusp::multiply(A, p, Ap);

			for (int j=0; j<recompute_r && j<(monitor.iteration_count()+1); ++j){

				ValueType beta = - blas::dotc(Ap, ApN[i]) / dotc_Ap[i];

				blas::axpy(pN[i], p, beta);

				blas::axpy(ApN[i], Ap, beta);

				if (i++ == recompute_r-1) i=0;

				if (j == monitor.iteration_count()) flag=false;

				if (j == (recompute_r-1)) flag=true;
			}

			blas::copy(p, pN[i]);

			blas::copy(Ap, ApN[i]);

			dotc_Ap[i] = blas::dotc(Ap, Ap);

			// alpha <- <r,Ap>/<Ap,Ap>
			alpha =  blas::dotc(r, Ap) / dotc_Ap[i];

			i++;

			++monitor;
		}
	}
	else
		throw cusp::invalid_input_exception("unrecognized method");
}

} // end namespace krylov
} // end namespace cusp

