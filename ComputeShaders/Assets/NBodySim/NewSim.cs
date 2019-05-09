using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections.LowLevel.Unsafe;
using System;
using UnityEngine.UI;

public class NewSim : MonoBehaviour {

    public ComputeShader NBodyCUDA;
    public Text test;

    #region new_variables
    public struct particle_t {
        public double x;
        public double y;
        public double vx;
        public double vy;
        public double ax;
        public double ay;
    };

    float density = 0.0005f;
    float mass = 0.01f;
    float cutoff = 0.01f;
    float min_r = 0.0001f;
    float dt = 0.0005f;

    [Tooltip("NUMBER OF PARTICLES")]
    public int n = 1000;

    private float size;

    const int NUM_THREADS = 256;
    const int NSTEPS = 1000; //1000
    ComputeBuffer d_particles;
    particle_t[] particles;

    ComputeBuffer d_particle_chain;
    ComputeBuffer d_bin;
    ComputeBuffer d_bin_debug;                                                                                                  //Debugging


    #endregion

    unsafe void Awake() {
        Debug.Log(Time.realtimeSinceStartup);
        size = (float)Math.Sqrt(density*n);
    }

    // Use this for initialization

    private void Start()
    {
        Debug.Log(Time.realtimeSinceStartup);
        StartCoroutine(nPartSim());
    }

    IEnumerator nPartSim() {
        Debug.Log("Starting NuStart");
        nuStart();
        yield return null;
    }


    unsafe public void nuStart () {

        float initTime = Time.realtimeSinceStartup;
        Debug.Log(Time.realtimeSinceStartup);

        d_particles = new ComputeBuffer(n, sizeof(particle_t)); // For GPU

        particles = new particle_t[n];  // Local stuff
    
        int sx = (int)Mathf.Ceil(Mathf.Sqrt((float)n));
        int sy = (n + sx - 1) / sx;

        int[] shuffle = new int[n];
        for (int i = 0; i < n; i++)
            shuffle[i] = i;

        for (int i = 0; i < n; i++)
        {
            //
            //  make sure particles are not spatially sorted
            //
            int j = UnityEngine.Random.Range(0, 2147483640) % (n - i);
            Debug.Log(j);
            int k = shuffle[j];
            shuffle[j] = shuffle[n - i - 1];

            //
            //  distribute particles evenly to ensure proper spacing
            //
            particles[i].x = size * (1.0f + (k % sx)) / (1 + sx);
            particles[i].y = size * (1.0f + (k / sx)) / (1 + sy);

            //
            //  assign random velocities within a bound
            //
            particles[i].vx = UnityEngine.Random.Range(0.0f, 1.0f) * 2 - 1;
            particles[i].vy = UnityEngine.Random.Range(0.0f, 1.0f) * 2 - 1;
        }

        d_particles.SetData(particles); 

        int particle_blks = (n + NUM_THREADS - 1) / NUM_THREADS;

        double bin_size = 2 * cutoff;
        double area_size = Mathf.Sqrt((float)density * n);
        int nbin_1d = (int)Mathf.Ceil(1.0f * (float)area_size / (float)bin_size);
        Vector2 bin_threads = new Vector2(Mathf.Sqrt(NUM_THREADS), Mathf.Sqrt(NUM_THREADS));
        int bin_blk = (int)Mathf.Ceil((float)(1.0 * nbin_1d / Mathf.Sqrt(NUM_THREADS)));
        Vector2 bin_blks = new Vector2(bin_blk, bin_blk);

        d_particle_chain = new ComputeBuffer(n, sizeof(int));
        d_bin = new ComputeBuffer(nbin_1d* nbin_1d, sizeof(int));
        d_bin_debug = new ComputeBuffer(n, sizeof(float));                    // Debugging

        NBodyCUDA.SetInt("nbin_1d", nbin_1d);

        //
        // Declare Constants for Compute Shaders
        //
        NBodyCUDA.SetFloat("density", density);
        NBodyCUDA.SetFloat("mass", mass);
        NBodyCUDA.SetFloat("cutoff", cutoff);
        NBodyCUDA.SetFloat("min_r", min_r);
        NBodyCUDA.SetFloat("dt", dt);
        NBodyCUDA.SetFloat("bin_size", 2*cutoff);
        NBodyCUDA.SetFloat("size", size);
        NBodyCUDA.SetInt("count", 11);

        //
        //  simulate a number of time steps
        //
        for (int step = 0; step < NSTEPS; step++)
        {
            int kID_clear_bin_gpu = NBodyCUDA.FindKernel("clear_bin_gpu");
            NBodyCUDA.SetInt("nbin_1d", nbin_1d);
            NBodyCUDA.SetBuffer(kID_clear_bin_gpu, "d_bin", d_bin);

            NBodyCUDA.Dispatch(kID_clear_bin_gpu, (int)bin_blks.x, (int)bin_blks.y, 1);

            int kID_assign_particle_gpu = NBodyCUDA.FindKernel("assign_particle_gpu");
            NBodyCUDA.SetInt("n", n);
            NBodyCUDA.SetBuffer(kID_assign_particle_gpu, "d_bin", d_bin);

            Shader.SetGlobalBuffer(Shader.PropertyToID("d_particles"), d_particles);
            Shader.SetGlobalBuffer(Shader.PropertyToID("d_particle_chain"), d_particle_chain);
            NBodyCUDA.Dispatch(kID_assign_particle_gpu , particle_blks, 1, 1);


            //
            //  compute forces
            //
            int kID_compute_forces_gpu = NBodyCUDA.FindKernel("compute_forces_gpu");
            NBodyCUDA.SetBuffer(kID_compute_forces_gpu, "d_bin", d_bin);
            Shader.SetGlobalBuffer(Shader.PropertyToID("d_bin_debug"), d_bin_debug);
           // NBodyCUDA.SetBuffer(kID_compute_forces_gpu, "d_bin_debug", d_bin_debug);
            NBodyCUDA.Dispatch(kID_compute_forces_gpu, particle_blks, 1, 1);

            //
            //  move particles
            //

            int kID_move_gpu = NBodyCUDA.FindKernel("move_gpu");
            

            NBodyCUDA.Dispatch(kID_move_gpu, particle_blks, 1, 1);

        }

        Debug.Log("Simulation over");
        float final = (Time.realtimeSinceStartup - initTime);
        test.text = final.ToString();

        Debug.Log(size);

        d_particles.Release();
        d_particle_chain.Release();
        d_bin.Release();
        d_bin_debug.Release();

    }

    
    private void GetMinDistance(particle_t[] final_particles)
    {
        Debug.Log("Started Distance Calculation");

        double min_distance = 10f;
        double sum = 0f;
        double n_new = 0f;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double temp = Math.Sqrt( Math.Pow((final_particles[i].x - final_particles[j].x), 2) + Math.Pow((final_particles[i].y - final_particles[j].y), 2));
                sum += temp;
                n_new++;
                if (temp < min_distance)
                {
                    min_distance = temp;
                }
            }
        }
        double avg = sum / n_new;
        Debug.Log("Started Distance Calculation. Min Distance is " + min_distance + " which is " + min_distance/0.02f + " of cutoff. Average is " + avg + " i.e " + avg/0.02f + " of cutoff.The size is " + size);

    }
	// Update is called once per frame
	void Update () {
		
	}
}
