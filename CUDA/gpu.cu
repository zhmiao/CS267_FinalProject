#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

extern double size;
//
//  benchmarking program
//

__global__ void clear_bin_gpu(int nbin_1d, int* d_bin)
{
  size_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  size_t tidy = threadIdx.y + blockIdx.y*blockDim.y;

  if(tidx < nbin_1d && tidy < nbin_1d) {
    d_bin[tidx*nbin_1d + tidy] = -1;
  }
}

__global__ void assign_particle_gpu(int n, particle_t* d_particles, double bin_size, int nbin_1d, int* d_bin, int* d_particle_chain)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < n) {
    particle_t* p_temp = &d_particles[tid];
    int nx = floor(p_temp->x/bin_size);
    int ny = floor(p_temp->y/bin_size);
    if(nx == nbin_1d) nx--;
    if(ny == nbin_1d) ny--;
    p_temp->ax = p_temp->ay = 0;
    d_particle_chain[tid] = atomicExch(&d_bin[nx*nbin_1d + ny], tid);
  }
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(int n, particle_t* d_particles, double bin_size, int nbin_1d, int* d_bin, int* d_particle_chain)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t* p_temp = &d_particles[tid];

  int nx = floor(p_temp->x/bin_size);
  int ny = floor(p_temp->y/bin_size);
  if(nx == nbin_1d) nx--;
  if(ny == nbin_1d) ny--;

  // Iterate through the particles in the same bin
  for(int i = d_bin[nx*nbin_1d + ny]; i != -1; i = d_particle_chain[i])
    if(i != tid) apply_force_gpu(*p_temp, d_particles[i]);
  // Iterate through the particles in neighboring bins
  if( nx == 0 && ny == 0 ) {
    for(int i = d_bin[(nx+1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  } else if( nx == 0 && ny == nbin_1d-1 ) {
    for(int i = d_bin[(nx+1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  } else if( nx == nbin_1d-1 && ny == 0 ) {
    for(int i = d_bin[(nx-1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  } else if(nx == nbin_1d-1 && ny == nbin_1d-1 ) {
    for(int i = d_bin[nx*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  } else if(nx == 0 ) {
    for(int i = d_bin[nx*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  } else if(nx == nbin_1d-1 ) {
    for(int i = d_bin[nx*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  } else if(ny == 0 ) {
    for(int i = d_bin[(nx+1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  } else if(ny == nbin_1d-1 ) {
    for(int i = d_bin[(nx+1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
   } else {
    for(int i = d_bin[nx*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[nx*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx-1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny-1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
    for(int i = d_bin[(nx+1)*nbin_1d + ny+1]; i != -1; i = d_particle_chain[i]) apply_force_gpu(*p_temp, d_particles[i]);
  }

}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}


int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    // GPU bin structure
    int *d_particle_chain;
    int *d_bin;

    //int particle_blks = ceil(n/NUM_THREADS);
    int particle_blks = (n + NUM_THREADS - 1)/NUM_THREADS;

    double bin_size = 2*cutoff;
    double area_size = sqrt( density * n );
    int nbin_1d = ceil(1.0*area_size/bin_size);
    dim3 bin_threads(sqrt(NUM_THREADS), sqrt(NUM_THREADS));
    int bin_blk = ceil(1.0*nbin_1d/sqrt(NUM_THREADS));
    dim3 bin_blks(bin_blk, bin_blk); 

    cudaMalloc((void **) &d_particle_chain, n * sizeof(int));
    cudaMalloc((void **) &d_bin, nbin_1d*nbin_1d*sizeof(int)); 

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        // Clear bins and assign particles to corresponding bin, reset acceleration
        clear_bin_gpu <<< bin_blks, bin_threads >>> (nbin_1d, d_bin);
	assign_particle_gpu <<< particle_blks, NUM_THREADS >>> (n, d_particles, bin_size, nbin_1d, d_bin, d_particle_chain);

        //
        //  compute forces
        //
	compute_forces_gpu <<< particle_blks, NUM_THREADS >>> (n, d_particles, bin_size, nbin_1d, d_bin, d_particle_chain);

        //
        //  move particles
        //
	move_gpu <<< particle_blks, NUM_THREADS >>> (d_particles, n, size);

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
