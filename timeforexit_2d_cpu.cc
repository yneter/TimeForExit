#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

#define ENABLE_OPENMP
#if defined(ENABLE_OPENMP)
#include<omp.h>
#endif 

static const int NR = 9;
static const int NTheta = 8;
static const int NFORCES = 30;

static const int MAXGENS = 2;
static const double PXOVER = 0.8;
static const double PMUTATION = 0.05;
static const int POPSIZE = NFORCES;
static const int NVARS = NR * NTheta;

#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

namespace odeint = boost::numeric::odeint;

typedef std::complex<double> point;
namespace point_util { 
template<typename T>
T &real(std::complex<T> &c) {
  return reinterpret_cast<T*>(&c)[0];
}

template<typename T>
T &imag(std::complex<T> &c) {
  return reinterpret_cast<T*>(&c)[1];
}
};
inline double get_x(const point &p) { return p.real(); }
inline double get_y(const point &p) { return p.imag(); }
inline double &ref_x(point &p) { return point_util::real<double>(p); }
inline double &ref_y(point &p) { return point_util::imag<double>(p); }
double myrand(void) { return (double) rand() / (double) RAND_MAX; }

typedef boost::array<double, 2> walk_state;

struct force_for_run {
    double *p;

    enum {X, Y};
  
    force_for_run(double *data) 
    {
        p = data;
    }

    double map(int nr, int ntheta) const { 
       return p[nr * NTheta + ntheta];
    }


    inline int bct(int j) const { return ( (j >= 0) ? j : j + NTheta) % NTheta; } 
    inline int bcr(int j) const { return (j < NR) ? j : NR-1; } 

    double ftheta_nr_ntheta(int ir, int it) const { 
       double dtheta = 2.0 * M_PI / (double) NTheta;
       double r = (double) (ir+0.5)/(double) NR;
       return -( map(bcr(ir), bct(it+1)) - map(bcr(ir), bct(it)) ) / (dtheta * r);
    }

    double fr_nr_ntheta(int ir, int it) const { 
       return -( map(bcr(ir+1), bct(it)) - map(bcr(ir), bct(it)) ) * (double) NR;
    }

    point force(point r) const { 
       double rr = abs(r); 
       double xr = abs(r) * (double) (NR-1);
       double xtheta = (arg(r)/(2.0 * M_PI)) * (double) NTheta ;
       
       int irl = (int) floor(xr);
       int irh = (int) ceil(xr); 
       int itl = (int) floor(xtheta);
       int ith = (int) ceil(xtheta);

       double prl = xr - (double) irl;
       double ptl = xtheta - (double) itl;
       double prh = 1.0 - prl;
       double pth = 1.0 - ptl;
       
       double fr = prl * ptl * fr_nr_ntheta(irl, itl) 
	         + prh * ptl * fr_nr_ntheta(irh, itl)
	         + prl * pth * fr_nr_ntheta(irl, ith)
	         + prh * pth * fr_nr_ntheta(irh, ith);
       
       
       double ft = prl * ptl * ftheta_nr_ntheta(irl, itl) 
	         + prh * ptl * ftheta_nr_ntheta(irh, itl)
	         + prl * pth * ftheta_nr_ntheta(irl, ith)
	         + prh * pth * ftheta_nr_ntheta(irh, ith);
              
       std::complex<double> uu = r / (rr + 0.5/(double) NR);
       return std::complex<double>(fr, ft) * uu;

    }

    void operator()(const walk_state &s, walk_state &dsdt, double t) const {
       point r(s[X], s[Y]);
       point f = force(r);
       dsdt[X] = get_x(f);
       dsdt[Y] = get_y(f);
    }
};


typedef std::mt19937 random_gen;

class time_for_exit { 
public : 
    double dt;
    double tmax;
    double D;
    double Umax;
    int nav;
    std::uniform_real_distribution<double> dis;

    time_for_exit(void) : dis(0.0, 1.0) { 
    }

    double run_for_exit(const force_for_run &w, random_gen &urand) { 
       odeint::runge_kutta4_classic<walk_state> stepper;
       point p = std::polar(1e-9, 2.0 * M_PI * dis(urand));
       walk_state s;
       s[0] = get_x(p); 
       s[1] = get_y(p);

       double t = 0;
       while (t < tmax) { 
	  stepper.do_step(w, s, 0, dt);
	  t += dt;
	  s[0] += sqrt(6.0 * D * dt) * (2.0 * dis(urand) - 1.0);
	  s[1] += sqrt(6.0 * D * dt) * (2.0 * dis(urand) - 1.0);
	  // for 1d like checks 
	  //	  s[1] = 0;
	  if ( s[0]*s[0] + s[1]*s[1] >= 1.0 ) {
	     break;
	  }
       }
       return t;
    }


    double mean_exit_time(const force_for_run &w, int seed = 123) { 
       random_gen urand(seed);
       double mean = 0;
       for (int i = 0; i < nav; i++) {
	  mean += run_for_exit(w, urand);
       }
       return mean /= (double) nav;
    }

    double upper(int i) const { 
       int nr = i / NTheta;
       return Umax * 2.0 * (double) std::min(nr, NR-1-nr) / (double) NR;
    }

    double lower(int i) const { 
       return -upper(i);
    }
   
    double score(double *raw_array, int seed = 123) { 
       force_for_run p(raw_array);
       return mean_exit_time(p, seed);
    }

    void print_info(void) { 
      std::cout << "# dt " << dt << std::endl;
      std::cout << "# tmax " << tmax << std::endl;
      std::cout << "# D " << D << std::endl;
      std::cout << "# Umax " << Umax << std::endl;
      std::cout << "# nav " << nav << std::endl;
    }
};



struct genotype
{
  double gene[NVARS];
  double score;
  double fitness;
  double upper[NVARS];
  double lower[NVARS];
  double rfitness;
  double cfitness;
};


template <class fitness_finder> class simple_GA { 
//    Modified simple GA 
//
//    Original version by Dennis Cormier/Sita Raghavan/John Burkardt.
//
//  Reference:
//
//    Zbigniew Michalewicz,
//    Genetic Algorithms + Data Structures = Evolution Programs,
//    Third Edition,
//    Springer, 1996,
//    ISBN: 3-540-60676-9,
//    LC: QA76.618.M53.
//
   fitness_finder &ffinder;
    
   int int_uniform_ab ( int a, int b ) { 
      return a + (rand() % (b - a + 1));
   }

   double real_uniform_ab ( double a, double b ) { 
      return a + (b - a) * (double) rand() / (double) RAND_MAX;
   }

   void Xover ( int one, int two ) {
      //  Select the crossover point.
      int point = int_uniform_ab ( 0, NVARS - 1 );
      //  Swap genes in positions 0 through POINT-1.
      for (int i = 0; i < point; i++ ) {
	 double t = population[one].gene[i];
	 population[one].gene[i] = population[two].gene[i];
	 population[two].gene[i] = t;
      }
   }

   void copy_gene(int from, int to) { 
      for (int i = 0; i < NVARS; i++ ) {
        population[to].gene[i] = population[from].gene[i];
      }
      population[to].score = population[from].score;
      population[to].fitness = population[from].fitness;      
   }

public : 
   struct genotype population[POPSIZE+1];
   struct genotype newpopulation[POPSIZE+1]; 
   double temp;

   simple_GA(fitness_finder &f) : ffinder(f) {
   }

   void crossover (void) {
      const double a = 0.0;
      const double b = 1.0;
      int mem;
      int one;
      int first = 0;
      
      for ( mem = 0; mem < POPSIZE; ++mem ) {
	double x = real_uniform_ab ( a, b );
	
	if ( x < PXOVER ) {
	  ++first;
	  
	  if ( first % 2 == 0 ) {
	    Xover ( one, mem );
	  } else {
	    one = mem;
	  }
	}
      }
      return;
   }

// 
//  If the best individual from the new population is better than 
//  the best individual from the previous population, then 
//  copy the best from the new population; else replace the 
//  worst individual from the current population with the 
//  best one from the previous generation                     
//  
   void elitist (void) {
     int i;
     double best, worst;
     int best_mem, worst_mem;
     
     best = worst = population[0].fitness;
     best_mem = worst_mem = 0;

     for (i = 0; i < POPSIZE - 1; ++i) {
        if ( population[i+1].fitness < population[i].fitness ) {
	   if ( best <= population[i].fitness ) {
	      best = population[i].fitness;
	      best_mem = i;
	   }
	   
	   if ( population[i+1].fitness <= worst ) {
	      worst = population[i+1].fitness;
	      worst_mem = i + 1;
	   }
	} else {
	  if ( population[i].fitness <= worst ) {
	     worst = population[i].fitness;
	     worst_mem = i;
	  }
	  if ( best <= population[i+1].fitness ) {
	     best = population[i+1].fitness;
	     best_mem = i + 1;
	  }
	}
     }

     if ( population[POPSIZE].fitness <= best ) {
        copy_gene(best_mem, POPSIZE); 
     } else {
        copy_gene(POPSIZE, worst_mem);
     } 
   }


   void evaluate ( void ) {
      // as we are dealing with random fitnesses we recompute the fitness of the best individual at POPSIZE 
      #pragma omp parallel for // num_threads(4)
      for (int member = 0; member <= POPSIZE; member++ ) { 
	 population[member].score = ffinder.score(population[member].gene);
      }
      double avscore = 0.0;
      for (int member = 0; member <= POPSIZE; member++ ) { 
	 avscore += population[member].score/(double) (POPSIZE+1);
      }
      for (int member = 0; member <= POPSIZE; member++ ) { 
	 population[member].fitness = exp ( -(population[member].score - avscore)/temp );
      }
   }

   void initialize (const double *init = NULL) {
      for (int j = 0; j <= POPSIZE; j++ ) {
         for (int i = 0; i < NVARS; i++ ) {
	    population[j].fitness = 0;	
	    population[j].score = 0;	
	    population[j].rfitness = 0;
	    population[j].cfitness = 0;
	    population[j].lower[i] = ffinder.lower(i);
	    population[j].upper[i] = ffinder.upper(i);
	    if (init == NULL) { 
	       if (j) { population[j].gene[i] = real_uniform_ab (population[j].lower[i], population[j].upper[i]); }
	       else { population[j].gene[i] = (population[j].lower[i] + population[j].upper[i]) / 2.0; }	      
	    } else { 
	       population[j].gene[i] = init[i];
	    }
	 }
      }
   }  

   void keep_the_best (void) { 
      int cur_best;
      int mem;
      int i;
      
      cur_best = 0;
      population[POPSIZE].fitness = 0;
      
      for ( mem = 0; mem < POPSIZE; mem++ ) {
        if ( population[POPSIZE].fitness < population[mem].fitness ) {
	  cur_best = mem;
	  population[POPSIZE].fitness = population[mem].fitness;
	}
      }
      // 
      //  Once the best member in the population is found, copy the genes.
      //
      copy_gene(cur_best, POPSIZE);
      return;
   }


   void mutate (void) { 
      const double a = 0.0;
      const double b = 1.0;
      double lbound;
      double ubound;
      double x;

      for (int i = 0; i < POPSIZE; i++ ) {
	 for (int j = 0; j < NVARS; j++ ) {
	    x = real_uniform_ab (a, b);
	    if ( x < PMUTATION ) {	      
	       lbound = population[i].lower[j];
	       ubound = population[i].upper[j];
	       population[i].gene[j] = real_uniform_ab (lbound, ubound);
	    }
	 }
      }
   }
  

   void mutate (double amplitude) { 
      const double a = 0.0;
      const double b = 1.0;
      double lbound;
      double ubound;
      double x;

      for (int i = 0; i < POPSIZE; i++ ) {
	 for (int j = 0; j < NVARS; j++ ) {
	    x = real_uniform_ab (a, b);
	    if ( x < PMUTATION ) {	      
	       lbound = std::max(population[i].lower[j], population[i].gene[j] - amplitude);
	       ubound = std::min(population[i].upper[j], population[i].gene[j] + amplitude);
	       population[i].gene[j] = real_uniform_ab (lbound, ubound);
	    }
	 }
      }
   }

   void report ( int generation ) {
      double avg;
      double best_val;
      double square_sum;
      double stddev;
      double sum;
      double sum_square;
      double av_score; 

      if ( generation == 0 ) {
	 std::cout << "\n";
	 std::cout << "Value     Generation    Best         Best       Average    Average    Standard \n";
	 std::cout << "Value     number        value        Score      fitness    score      deviation \n";
	 std::cout << "\n";
      }

      sum = 0.0;
      sum_square = 0.0;
      av_score = 0.0;

      for (int i = 0; i < POPSIZE; i++ ) {
	 sum += population[i].fitness;
	 sum_square += population[i].fitness * population[i].fitness;
	 av_score += population[i].score;
      }

      avg = sum / ( double ) POPSIZE;
      av_score /= (double) POPSIZE;
      square_sum = avg * avg * POPSIZE;
      stddev = sqrt ( ( sum_square - square_sum ) / ( POPSIZE - 1 ) );
      best_val = population[POPSIZE].fitness;
      double best_score = population[POPSIZE].score;

      std::cout << "  " << std::setw(8) << "equal " 
                << "  " << std::setw(8) << generation 
		<< "  " << std::setw(14) << best_val 
		<< "  " << std::setw(14) << best_score
		<< "  " << std::setw(14) << avg 
		<< "  " << std::setw(14) << av_score	
		<< "  " << std::setw(14) << stddev << "\n";

   }

 
   void selector (void) {
      const double a = 0.0;
      const double b = 1.0;
      int i;
      int j;
      int mem;
      double p;
      double sum;
      //
      //  Find the total fitness of the population.
      //
      sum = 0.0;
      for ( mem = 0; mem < POPSIZE; mem++ ) {
	 sum = sum + population[mem].fitness;
      }
      //
      //  Calculate the relative fitness of each member.
      //
      for ( mem = 0; mem < POPSIZE; mem++ ) {
	 population[mem].rfitness = population[mem].fitness / sum;
      }
      // 
      //  Calculate the cumulative fitness.
      //
      population[0].cfitness = population[0].rfitness;
      for ( mem = 1; mem < POPSIZE; mem++ ) {
	 population[mem].cfitness = population[mem-1].cfitness + population[mem].rfitness;
      }
      // 
      //  Select survivors using cumulative fitness. 
      //
      for ( i = 0; i < POPSIZE; i++ ) { 
	 p = real_uniform_ab (a, b);
	 if ( p < population[0].cfitness ) {
	    newpopulation[i] = population[0];      
	 } else {
	    // could use a dichotomic search - 
	    for ( j = 0; j < POPSIZE; j++ ) { 
	       if ( population[j].cfitness <= p && p < population[j+1].cfitness ) {
		  newpopulation[i] = population[j+1]; 
	       }
	    }
	 }
      }
      // 
      //  Overwrite the old population with the new one.
      //
      for ( i = 0; i < POPSIZE; i++ ) {
	 population[i] = newpopulation[i]; 
      }

      return;     
   }

   void print_info(void) { 
      std::cout << "# POPSIXE " << POPSIZE << std::endl;
      std::cout << "# MAXGENS " << MAXGENS << std::endl;
      std::cout << "# NVARS " << NVARS << std::endl;
      std::cout << "# PXOVER " << PXOVER << std::endl;
      std::cout << "# PMUTATION " << PMUTATION << std::endl;
      std::cout << "# temp " << temp << std::endl;
   }

   void print_best(void) { 
      std::cout << "# best gene = " << population[POPSIZE].fitness << "\n";
      for (int i = 0; i < NVARS; i++ ) {
	 std::cout << i << "    " << population[POPSIZE].gene[i] << "  %" << std::endl;
      }
      std::cout << "# with fitness = " << population[POPSIZE].fitness << "\n";
   }
};


int main(int argc, char *argv[]) {
    std::cout << "# NR " <<  NR << std::endl;
    std::cout << "# NTheta " <<  NTheta << std::endl;



    srand(50);
    double zero_potential[NR * NTheta];
    for (int i = 0; i < NR * NTheta; i++) zero_potential[i] = 0.0;

    /***    
    double sym_potential_r[NR] = { 0.0, 0.00792976, 0.0366886, 0.046863, 0.0749836, 0.053039, 0.0163167, -0.0220963 };
    double sym_potential[NR * NTheta];
    for (int r = 0; r < NR; r++) { 
       for (int t = 0; t < NTheta; t++) { 
	  sym_potential[r * NTheta + t] = sym_potential_r[r];
       }
    }
    ***/

    force_for_run p0(zero_potential);
    time_for_exit tex;
    simple_GA<time_for_exit> ga(tex);

    double anneal_eps = 3.0e-3;
    std::cout.setf(std::ios::unitbuf);

    tex.D = 0.1;
    tex.tmax = 100.0;
    tex.dt = 0.01;
    tex.nav = 1000;
    tex.Umax = 0.3;    
    tex.print_info();

    ga.temp = 0.5 * tex.mean_exit_time(p0); 
    double temp_min = 0.1 * ga.temp;
    int nav_max = 100000;
    std::cout << "# temp_min " << temp_min << std::endl;
    std::cout << "# nav_max " << nav_max << std::endl;
    //    ga.initialize(sym_potential);
    ga.initialize();
    std::cout << "# anneal_eps " << anneal_eps << std::endl;
    std::cout << "# ga initialized "  << std::endl;

    ga.print_info();
    ga.evaluate ();
    std::cout << "# ga evaluate "  << std::endl;
    ga.report(-1);
    ga.keep_the_best();

    for (int generation = 0; generation < MAXGENS; generation++ ) {
       ga.temp *= (1.0 - anneal_eps);
       if (ga.temp < temp_min) ga.temp = temp_min;
       tex.nav *= (1.0 + anneal_eps);
       if (tex.nav > nav_max) { tex.nav = nav_max; }
       ga.selector ();
       ga.crossover ();
       ga.mutate ();
       ga.report(generation);
       ga.evaluate ();
       ga.elitist();
       if (!(generation % 20)) {
	  ga.print_best();	  
       }
    }
    ga.print_best();
    return 0;
}
