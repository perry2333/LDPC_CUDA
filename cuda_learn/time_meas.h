#ifdef OMP
#include <omp.h>
#endif

typedef struct {

  long long in;
  long long diff;
  long long max;
  int trials;
} time_stats_t;

static inline void start_meas(time_stats_t *ts) __attribute__((always_inline));
static inline void stop_meas(time_stats_t *ts) __attribute__((always_inline));

#if defined(__i386__)
static inline unsigned long long rdtsc_oai(void) __attribute__((always_inline));
static inline unsigned long long rdtsc_oai(void) {
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}
#elif defined(__x86_64__)
static inline unsigned long long rdtsc_oai() __attribute__((always_inline));
static inline unsigned long long rdtsc_oai() { 
  unsigned long long a, d;
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
  return (d<<32) | a;
}
#elif defined(__aarch64__)
static inline uint64_t rdtsc_oai(void) __attribute__((always_inline));
static inline uint64_t rdtsc_oai(void)
{
  uint64_t r = 0;
  asm volatile("mrs %0, cntvct_el0" : "=r"(r));
  return r;
}

#elif defined(__arm__)
static inline uint32_t rdtsc_oai(void) __attribute__((always_inline));
static inline uint32_t rdtsc_oai(void) {
  uint32_t r = 0;
  asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(r) );
  return r;
}
#endif

static inline void start_meas(time_stats_t *ts) {

#ifdef OMP
  int tid;

  tid = omp_get_thread_num();
  if (tid==0)
#endif
    {
      ts->trials++;
      ts->in = rdtsc_oai();
    }
}

static inline void stop_meas(time_stats_t *ts) {

  long long out = rdtsc_oai();

#ifdef OMP
  int tid;
  tid = omp_get_thread_num();
  if (tid==0)
#endif
    {
      ts->diff += (out-ts->in);
      if ((out-ts->in) > ts->max)
	    ts->max = out-ts->in;
      
    }
}

static inline void reset_meas(time_stats_t *ts) {

  ts->trials=0;
  ts->diff=0;
  ts->max=0;
}
