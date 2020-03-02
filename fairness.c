/* Copyright 2015 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <sched.h>
#include <string.h>

#include "cpu_util.h"
#include "expand.h"
#include "timer.h"
#include "stats.h"
#ifndef COUNT_SWEEP_MAX
#define COUNT_SWEEP_MAX 32
#endif
#ifndef NUM_COUNTERS
#define NUM_COUNTERS 256
#endif
#define MAX_STATS 4

char *stat_desc[MAX_STATS] = {"Acquires","NFails","NBusy","NMisc"};
static char full_name[80];

typedef unsigned atomic_t;

typedef union {
        struct {
            atomic_t count;
            int cpu;
			int counter;
			int mode;
        } x;
		atomic_t stats[MAX_STATS];
        char pad[AVOID_FALSE_SHARING];
} per_thread_t;

typedef struct {
		int lock;
		int hold;
        struct {
                atomic_t count;
        } x[NUM_COUNTERS];
        char pad1[CACHELINE_SIZE];
		pthread_mutex_t mutex[NUM_COUNTERS];
        char pad2[CACHELINE_SIZE];
} global_t;

//Multiple counters to test performance under false sharing conditions.
global_t global_counter[COUNT_SWEEP_MAX];

static volatile int relaxed;
static volatile int count_sweep=0;
static volatile int sweep_active=1;
static size_t req_threads=0;


static pthread_mutex_t wait_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t wait_cond = PTHREAD_COND_INITIALIZER;
static size_t nr_to_startup;
static uint64_t delay_mask;
static int inc_count=1;

static void wait_for_startup(void)
{
        // wait for everyone to spawn
        pthread_mutex_lock(&wait_mutex);
        --nr_to_startup;
        if (nr_to_startup) {
			pthread_cond_wait(&wait_cond, &wait_mutex);
        }
        else {
			pthread_cond_broadcast(&wait_cond);
        }
        pthread_mutex_unlock(&wait_mutex);
}

void __attribute__((noinline, optimize("no-unroll-loops"))) blackhole(unsigned long iters);

void __attribute__((noinline, optimize("no-unroll-loops"))) blackhole(unsigned long iters) {
    if (! iters) { return; }
#ifdef __aarch64__
    asm volatile (".p2align 4; 1: add %0, %0, -1; cbnz  %0, 1b" : "+r" (iters) : "0" (iters));
#elif __x86_64__
    asm volatile (".p2align 4; 1: add $-1, %0; jne 1b" : "+r" (iters) );
#else
	#error "Don't have asm version for work section for this architecture"
#endif
}

static inline void work_section(unsigned long count, int tmp) {
	blackhole(count);
}


static volatile int first=1;
static void test_broadcast(void)
{
        // wait for everyone to spawn
        pthread_mutex_lock(&wait_mutex);
		if (first) {
			first=0;
			nr_to_startup = req_threads;
		}
        --nr_to_startup;
        if (nr_to_startup) {
			pthread_cond_wait(&wait_cond, &wait_mutex);
        }
        else {
			nr_to_startup = req_threads;
			pthread_cond_broadcast(&wait_cond);
        }
        pthread_mutex_unlock(&wait_mutex);
}


static inline void test_cond(per_thread_t *args) {
	unsigned long parallel_time=global_counter[count_sweep].hold;
	volatile int tmp=0;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	while (sweep_active) {
		int i;
		for (i=0; i<inc_count ; i++) {
			test_broadcast();
			work_section(parallel_time,tmp);
		}
		__sync_fetch_and_add(&args->stats[0], i);
	}
}


static inline void test_mutex_lock_anemic(per_thread_t *args) {
	int cid=args->x.counter;
	unsigned long hold_time=global_counter[count_sweep].hold;
	volatile int tmp=0;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	pthread_mutex_t *p=&(global_counter[count_sweep].mutex[cid]);
	while (sweep_active) {
		int i,fails=0;
		for (i=0; i<inc_count ; i++) {
			int ret=pthread_mutex_lock(p);
			if (ret==0)
				pthread_mutex_unlock(p);
			else
				fails++;
			work_section(hold_time,tmp);
		}
		__sync_fetch_and_add(&args->stats[0], i);
		__sync_fetch_and_add(&args->stats[1], fails);
	}
}


static inline void test_mutex_trylock_anemic(per_thread_t *args) {
	int cid=args->x.counter;
	unsigned long hold_time=global_counter[count_sweep].hold;
	volatile int tmp=0;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	pthread_mutex_t *p=&(global_counter[count_sweep].mutex[cid]);
	while (sweep_active) {
		int busy = 0, fails=0, acquired=0, i;
		
		for (i=0; i<inc_count ; i++) {
			int ret = pthread_mutex_trylock(p);
			if (ret == EBUSY) { // Simulate skipping the lock
				busy++;
				continue;
			} 
			if (ret != 0) {
				fails++;
				continue;
			}
			// Otherwise, lock acquired do critical section
			pthread_mutex_unlock(p);
			acquired++;
			work_section(hold_time,tmp);
		}
		__sync_fetch_and_add(&args->stats[0], acquired);
		__sync_fetch_and_add(&args->stats[1], fails);
		__sync_fetch_and_add(&args->stats[2], busy);
	}
}


static inline void test_mutex_trylock(per_thread_t *args) {
	int cid=args->x.counter;
	unsigned long hold_time=global_counter[count_sweep].hold;
	unsigned long lock_time=global_counter[count_sweep].lock;
	volatile int tmp=0;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	pthread_mutex_t *p=&(global_counter[count_sweep].mutex[cid]);
	while (sweep_active) {
		int busy = 0, fails=0, acquired=0, i;
		
		for (i=0; i<inc_count ; i++) {
			int ret = pthread_mutex_trylock(p);
			if (ret == EBUSY) { // Simulate skipping the lock
				work_section(hold_time,tmp);
				busy++;
				continue;
			} 
			if (ret != 0) {
				fails++;
				continue;
			}
			// Otherwise, lock acquired do critical section
			work_section(lock_time,tmp);
			pthread_mutex_unlock(p);
			acquired++;
			work_section(hold_time,tmp);
		}
		__sync_fetch_and_add(&args->stats[0], acquired);
		__sync_fetch_and_add(&args->stats[1], fails);
		__sync_fetch_and_add(&args->stats[2], busy);
	}
}


static inline void test_malloc(per_thread_t *args) {
	int cid=args->x.counter;
	unsigned long block_work=global_counter[count_sweep].hold;
	unsigned long block_size=global_counter[count_sweep].lock;
	unsigned long block_count = inc_count;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	char **block = (char **)malloc(block_count * sizeof(char *));
	int rounds = 0;
	while (sweep_active) {
		unsigned long i,j,tmp=0;
		
		for (i=0; i<block_count ; i++) {
			block[i] = (char *)malloc(block_size * sizeof(char));
			for (j=0 ; j<block_work; j++) {
				block[i][j*64 % block_size] ^= cid++; 				
			}
		}
		tmp += block[cid % block_count][rounds++ % block_size];
		for (i=0; i<block_count ; i++) {
			free(block[i]);
		}
		__sync_fetch_and_add(&args->stats[0], block_count);
		__sync_fetch_and_add(&args->stats[1], tmp);
	}
}


static void *null_thread(void *vp) {
	unsigned long long *arg = (unsigned long long *)vp;
	unsigned long long null_hold = global_counter[count_sweep].hold;
	work_section(null_hold, 0);
	*arg = null_hold;
	pthread_exit(vp);
	return NULL;
}

/*
	Test thread creation latencies, allow created thread to do some work.
*/
static inline void test_thread_create(per_thread_t *args) {
	unsigned long thread_count = inc_count;
	unsigned long long null_hold = global_counter[count_sweep].hold;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	pthread_t *tag = (pthread_t *)malloc(sizeof(pthread_t) * thread_count);
	unsigned long long *retvals   = (unsigned long long *)malloc(sizeof(unsigned long long) * thread_count);
	int C=0;
	while (sweep_active) {
		unsigned long i;
		int created=0, ok=0, fails=0, incomplete=0;
		C++;
		for (i=0; i<thread_count ; i++) {
			retvals[i] = 0;
			tag[i] = 0;
			int err = pthread_create(&tag[i], NULL, null_thread, &retvals[i]);
			if (err == 0) {
				created++;
			} else {
				fails++;
				tag[i] = 0;
			}
		}
		for (i=0; i<thread_count ; i++) {
			unsigned long long *ret=0;
			if (tag[i] != 0) {
				pthread_join(tag[i], (void **)&ret);
				tag[i]=0;
				if (*ret == null_hold) {
					ok++;
				} else {
					incomplete++;
				}
			}
		}
		__sync_fetch_and_add(&args->stats[0], created);
		__sync_fetch_and_add(&args->stats[1], fails);
		__sync_fetch_and_add(&args->stats[2], ok);
		__sync_fetch_and_add(&args->stats[3], incomplete);
	}
}


unsigned long block_sizes[] = {5, 21, 32, 55, 256, 12, 8, 128};

static inline void test_malloc_rand(per_thread_t *args) {
	int cid=args->x.counter;
	unsigned long block_work=global_counter[count_sweep].hold;
	unsigned long block_count = inc_count;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	char **block = (char **)calloc(block_count , sizeof(char *));
	unsigned long *last_size = (unsigned long *)calloc(block_count , sizeof(unsigned long));
	int round = 0;
	while (sweep_active) {
		unsigned long i,j,tmp=0;
		
		for (i=0; i<block_count ; i++) {
			unsigned long block_size = block_sizes[(i^round) % sizeof(block_sizes)/sizeof(unsigned long)];
			last_size[i]=block_size;
			block[i] = (char *)malloc(block_size * sizeof(char));
			for (j=0 ; j<block_work; j++) {
				block[i][(j*64) % block_size] ^= cid++; 				
			}
		}
		unsigned int id = cid % block_count;
		tmp += block[id][round++ % last_size[id]];
		for (i=0; i<block_count ; i++) {
			free(block[i]);
		}
		__sync_fetch_and_add(&args->stats[0], block_count);
		__sync_fetch_and_add(&args->stats[1], tmp);
	}
	free(last_size);
}


static inline void test_malloc_frag(per_thread_t *args) {
	int cid=args->x.counter;
	unsigned long block_work=global_counter[count_sweep].hold;
	unsigned long block_count = inc_count;
	if (delay_mask & (1u<<args->x.cpu)) {
			sleep(1);
	}
	char **block = (char **)calloc(block_count, sizeof(char *));
	unsigned long *last_size = (unsigned long *)calloc(block_count , sizeof(unsigned long));
	int round = 0;
	while (sweep_active) {
		unsigned long i,j,tmp=0;
		// Allocate random block sizes
		for (i=0; i<block_count ; i++) {
			unsigned long block_size = block_sizes[(i^round) % sizeof(block_sizes)/sizeof(unsigned long)];
			last_size[i]=block_size;
			block[i] = (char *)malloc(block_size * sizeof(char));
			for (j=0 ; j<block_work; j++) {
				block[i][j*64 % block_size] ^= cid++; 				
			}
		}
		unsigned int id = cid % block_count;
		tmp += block[id][round++ % last_size[id]];
		unsigned int freed_blocks=0;
		// Free blocks in random order to force some fragmentation
		while (freed_blocks < block_count / 2) {
			i = tmp % block_count;
			while (block[i] == NULL) { // Find a block to free
				i = (i+1) % block_count;
			}
			tmp += (unsigned long)(block[i]) & 3;
			free(block[i]);
			freed_blocks++;
			block[i]=NULL;
		}
		__sync_fetch_and_add(&args->stats[0], freed_blocks);
		freed_blocks=0;
		// Now free the rest of the blocks
		for (i=0; i<block_count ; i++) {
			if (block[i] != NULL) { // Find a block to free
				free(block[i]);;
				freed_blocks++;
			}
			block[i] = NULL;
		}
		__sync_fetch_and_add(&args->stats[0], freed_blocks);
		__sync_fetch_and_add(&args->stats[1], tmp);
	}
	free(last_size);
}


static inline void test_mutex_lock(per_thread_t *args) {
	int cid=args->x.counter;
	unsigned long lock_time=global_counter[count_sweep].lock;
	unsigned long hold_time=global_counter[count_sweep].hold;
	volatile int tmp=0;
	while (sweep_active) {
		pthread_mutex_t *p=&(global_counter[count_sweep].mutex[cid]);
		int i;
		if (delay_mask & (1u<<args->x.cpu)) {
				sleep(1);
		}
		for (i=0; i<inc_count ; i++) {
			pthread_mutex_lock(p);
			work_section(lock_time,tmp);
			pthread_mutex_unlock(p);
			work_section(hold_time,tmp);
		}
		__sync_fetch_and_add(&args->stats[0], i);
	}
}

static inline void test_sync_fetch_and_add(per_thread_t *args) {
	int cid=args->x.counter;
	while (sweep_active) {
		atomic_t *p=&(global_counter[count_sweep].x[cid].count);
		int i;
		if (delay_mask & (1u<<args->x.cpu)) {
				sleep(1);
		}
		while (!relaxed) {
				for (i=0; i<inc_count ; i++) {
					x50(__sync_fetch_and_add(p, 1););
				}
				__sync_fetch_and_add(&args->stats[0], 50*i);

		}

		if (delay_mask & (1u<<args->x.cpu)) {
				sleep(1);
		}
		while (relaxed) {
				for (i=0; i<inc_count ; i++) {
					x50(__sync_fetch_and_add(p, 1); cpu_relax(););
				}
				__sync_fetch_and_add(&args->stats[0], 50*i);
		}
	}
	
}

static inline void test_atomic_fetch_and_add(per_thread_t *args) {
	int cid=args->x.counter;
	while (sweep_active) {
		atomic_t *p=&(global_counter[count_sweep].x[cid].count);
		int i;
		if (delay_mask & (1u<<args->x.cpu)) {
				sleep(1);
		}
		while (!relaxed) {
				for (i=0; i<inc_count ; i++) {
					x50(__atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST););
				}
				__sync_fetch_and_add(&args->stats[0], 50*i);
		}

		if (delay_mask & (1u<<args->x.cpu)) {
				sleep(1);
		}
		while (relaxed) {
				for (i=0; i<inc_count ; i++) {
					x50(__atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); cpu_relax(););
				}
				__sync_fetch_and_add(&args->stats[0], 50*i);
		}
	}
	
}

char *test_name[] = {
	"Sync fetch and add",
	"Atomic fetch and add",
	"pthread_mutex lock",
	"pthread_mutex trylock",
	"malloc",
	"malloc Rand",
	"malloc Frag",
	"pthread_cond",
	"pthread_create"
};


static inline char *get_test_name(int mode, int lock_time, int captive) {
	char *top = test_name[mode];
	char *tag = "";
	if (lock_time == 0 && strstr(top,"malloc") != NULL) {
		tag = " anemic";
	}
	if (captive > 0 && strstr(top,"mutex") != NULL ) {
		tag = " captive";
	}
	sprintf(full_name,"TEST,%s%s\n",top, tag);
	return full_name;
}


static void *worker(void *_args)
{
        per_thread_t *args = _args;
        
        // move to our target cpu
        cpu_set_t cpu;
        CPU_ZERO(&cpu);
        CPU_SET(args->x.cpu, &cpu);
        if (sched_setaffinity(0, sizeof(cpu), &cpu)) {
                perror("sched_setaffinity");
                exit(1);
        }
		unsigned long lock_time=global_counter[0].lock;

        wait_for_startup();

		switch (args->x.mode) {
			case 1:
				test_atomic_fetch_and_add(args);
				break;
			case 2:
				if (lock_time > 0)
					test_mutex_lock(args);
				else
					test_mutex_lock_anemic(args);
				break;
			case 3:
				if (lock_time > 0)
					test_mutex_trylock(args);
				else
					test_mutex_trylock_anemic(args);
				break;
			case 4:
				stat_desc[0] = "Blocks";
				test_malloc(args);
				break;
			case 5:
				stat_desc[0] = "Blocks";
				test_malloc_rand(args);
				break;
			case 6:
				stat_desc[0] = "Blocks";
				test_malloc_frag(args);
				break;
			case 7:
				test_cond(args);
				break;
			case 8:
				stat_desc[0] = "Threads";
				stat_desc[1] = "NFails";
				stat_desc[2] = "NComplete";
				stat_desc[3] = "NIncomplete";
				test_thread_create(args);
				break;
			default:
				test_sync_fetch_and_add(args);
				break;
		}
        return NULL;
}

int main(int argc, char **argv)
{
    int c,sweep_count=1;
	size_t max_samples=6;
	int sample_interval=500000;
	int verbosity=0, mode=0, lock_time=0, hold_time=0, max_relax=2, mask_uniq=1;
	static double spacer=0.;
	int used_stats=1;
	int captive_mutex=0;

	delay_mask = 0;
	while ((c = getopt(argc, argv, "d:s:n:t:v:N:r:m:l:h:i:T:M:")) != -1) {
			switch (c) {
			case 'd':
					delay_mask = strtoul(optarg, 0, 0);
					break;
			case 's':
					sweep_count = strtoul(optarg, 0, 0);
					break;
			case 'n':
					max_samples = strtoul(optarg, 0, 0);
					break;
			case 'T':
					req_threads = strtoul(optarg, 0, 0);
					break;
			case 'M':
					captive_mutex = strtoul(optarg, 0, 0);
					break;
			case 'm':
					mode = strtoul(optarg, 0, 0);
					if (mode>1) {
						max_relax=1;
						used_stats = 4;
					}
					break;
			case 'l':
					lock_time = strtoul(optarg, 0, 0);
					break;
			case 'i':
					mask_uniq = strtoul(optarg, 0, 0);
					break;
			case 'h':
					hold_time = strtoul(optarg, 0, 0);
					break;
			case 'v':
					verbosity = strtoul(optarg, 0, 0);
					break;
			case 'N':
					inc_count = strtoul(optarg, 0, 0);
					break;
			case 'r':
					spacer = strtod(optarg, NULL);
					break;
			case 't':
					sample_interval = 1000*strtoul(optarg, 0, 0);
					break;
			default:
					goto usage;
			}
	}

	if (argc - optind != 0) {
usage:
			fprintf(stderr, "usage: %s [-d delay_mask]\n"
							"by default runs one thread on each cpu, use taskset(1) to\n"
							"restrict operation to fewer cpus/threads.\n"
							"the optional delay_mask specifies a mask of cpus on which to delay\n"
							"the startup.\n"
							"-n : set number of samples\n"
							"-m : set mode (1=atomic, 2=mutex, 3=trylock, 4=malloc)\n"
							"-l : lock time for mutex, block size for malloc\n"
							"-h : work (hold parallel iterations for mutex, iterations over memory for malloc)\n"
							"-T : threads to create\n"
							"-N : number of loop iterations for mutex, or number of blocks for malloc\n"
							"-t : sampling interval\n"
							"-v : verbosity for extra stats\n"
							"-M : Hold mutex captive (for trylock)\n"
							, argv[0]);
			exit(1);
	}

	setvbuf(stdout, NULL, _IONBF, BUFSIZ);
	char *myname = get_test_name(mode, lock_time, captive_mutex);
	printf("%s", myname);
	if (strstr(myname, "mutex") || strstr(myname, "cond")) {
		printf("lock time,%d\n", lock_time);
		printf("hold time,%d\n", hold_time);
		printf("iterations per update,%d\n", inc_count);
	}
	if (strstr(myname, "malloc")) {
		printf("block size,%d\n", lock_time);
		printf("work per block,%d\n", hold_time);
		printf("number of blocks,%d\n", inc_count);
	}
	if (strstr(myname, "pthread_create")) {
		printf("number of dummy threads,%d\n", inc_count);
		printf("work per dummy thread,%d\n", hold_time);
	}
	printf("Threads %ld\n", req_threads);
	
	

	// find the active cpus
	cpu_set_t cpus;
	if (sched_getaffinity(getpid(), sizeof(cpus), &cpus)) {
			perror("sched_getaffinity");
			exit(1);
	}

	// could do this more efficiently, but whatever
	size_t nr_threads = 0;
	int i,j;
	for (i = 0; i < CPU_SETSIZE; ++i) {
			if (CPU_ISSET(i, &cpus)) {
					++nr_threads;
			}
	}
	for (i=0; i<COUNT_SWEEP_MAX; i++) { 
		global_counter[i].lock=lock_time;
		global_counter[i].hold=hold_time;
		
		for (j=0; j<NUM_COUNTERS; j++) {
			pthread_mutex_init(&global_counter[i].mutex[j],NULL);
			if (captive_mutex > 0) {
				pthread_mutex_lock(&global_counter[i].mutex[j]);
			}
		}
	}
	

	if (req_threads == 0) {
		req_threads=nr_threads;
	} 
	per_thread_t *thread_args = calloc(req_threads, sizeof(*thread_args));
	nr_to_startup = req_threads + 1;
	size_t u;
	int q = 0;
	for (u = 0; u < req_threads; ++u) {
			while (!CPU_ISSET(q, &cpus)) {
				q = (q+1) % (CPU_SETSIZE-1);
			}
			thread_args[u].x.cpu = q;
			if (mask_uniq > 1)
				thread_args[u].x.counter = (int)((double)u * spacer) % mask_uniq;
			else
				thread_args[u].x.counter = 0;
			q = (q+1) % (CPU_SETSIZE-1);
			thread_args[u].x.count = 0;
			thread_args[u].x.mode=mode;
			pthread_t dummy;
			if (pthread_create(&dummy, NULL, worker, &thread_args[u])) {
					perror("pthread_create");
					exit(1);
			}
	}

	wait_for_startup();

	atomic_t *samples = calloc(req_threads, sizeof(*samples) * MAX_STATS);

	printf("results are avg latency per locked increment in ns, one column per thread\n");
	printf("cpu,");
	for (u = 0; u < req_threads; ++u) {
			printf("%lu[%u],", u, thread_args[u].x.cpu);
	}
	printf("avg,stdev,min,max\n");
	char msg[256];
	stat_t stats[2][COUNT_SWEEP_MAX][MAX_STATS];
	stat_t global_stats[2];
	sprintf(msg,"Unrelaxed summary across %d global counts [latency avg in ns, bw in ops/mSec]",COUNT_SWEEP_MAX);
	stat_init(&global_stats[0],msg);
	sprintf(msg,"Relaxed summary across %d global counts [latency avg in ns, bw in ops/mSec]",COUNT_SWEEP_MAX);
	stat_init(&global_stats[1],msg);
	if (sweep_count > COUNT_SWEEP_MAX)
		sweep_count = COUNT_SWEEP_MAX;
    for (count_sweep = 0; count_sweep < sweep_count; ++count_sweep) {
		for (relaxed = 0; relaxed < max_relax; ++relaxed) {
			sprintf(msg,"Global counter %d %s [bw in ops/mSec per thread, latency in ns]",count_sweep,relaxed ? "relaxed:" : "unrelaxed:");
			printf("%s\n",msg);
			for (j=0; j < used_stats ; j++) {
				stat_init(&stats[relaxed][count_sweep][j],msg);
			}

			uint64_t last_stamp = now_nsec();
			size_t sample_nr;
			double bw=0.;
			for (sample_nr = 0; sample_nr < max_samples; ++sample_nr) {
				usleep(sample_interval);
				for (u = 0; u < req_threads; ++u) {
					for (j=0; j < used_stats ; j++) {
						samples[u*MAX_STATS+j] = __sync_lock_test_and_set(&thread_args[u].stats[j], 0);
						//samples[u] = __sync_lock_test_and_set(&thread_args[u].x.count, 0);
					}
				}
				uint64_t stamp = now_nsec();
				int64_t time_delta = stamp - last_stamp;
				last_stamp = stamp;

				// throw away the first sample to avoid race issues at startup / mode switch
				if (sample_nr == 0) continue;

				for (j=0; j < used_stats ; j++) { // for each stat aggregate and print data for all threads
					char line[128]="";
					char *p=line;
					double sum = 0.;
					double sum_squared = 0.;
					double max=0.0,min=1e100;
					int nval=0;
					int report_bw = (stat_desc[j][0] == 'N') ? 1 : 0;
					
					
					p+=sprintf(p,"%s",stat_desc[j]);
					for (u = 0; u < req_threads; ++u) {
						double val = (double)samples[u*MAX_STATS+j];
						if (val>0)
							nval++;
						double s = time_delta / val;
						//Get BW stats in ops per msec instead of per ns.
						double bwt = val * 1000000.0 / (double)time_delta;
						if (j==0) // for stat 0, accumulate bw for all threads
							bw += bwt; 
						if (report_bw) 
							s=bwt;
						if (min > s) min=s;
						if (max < s) max=s;
						p+=sprintf(p,",%.1f", s);
						sum += s;
						sum_squared += s*s;
						stat_add(&stats[relaxed][count_sweep][j],s,bwt);
					}
					p+=sprintf(p,",%.1f,%.1f,%.1f,%.1f\n",
							sum / req_threads,
							sqrt((sum_squared - sum*sum/req_threads)/(req_threads-1)),
							min,max);
					if (nval>0)
						printf("%s",line);
				}
			}
			for (j=0; j < used_stats ; j++) {
				stat_update(&stats[relaxed][count_sweep][j]);
			}
			stat_add(&global_stats[relaxed],
				stats[relaxed][count_sweep][0].latency.avg,
				bw/(double)sample_nr);
		}
	}
	sweep_active=0;
	if (captive_mutex > 0) {
		pthread_mutex_unlock(&global_counter[i].mutex[j]);
	}
	
	if (verbosity > 0) { 
		for (relaxed = 0; relaxed < max_relax; ++relaxed) {
			for (j=0; j < used_stats ; j++) {
				stat_print(&stats[relaxed][0][j]);
			}
		}
		if (verbosity > 1) {
			for (count_sweep = 1; count_sweep < sweep_count; ++count_sweep) {
				stat_print(&stats[0][count_sweep][0]);
			}
		}
		for (relaxed = 0; relaxed < max_relax; ++relaxed) 
			if ((sweep_count > 1) && (verbosity > 0) ) {
				stat_update(&global_stats[relaxed]);
				stat_print(&global_stats[relaxed]);
			}
	}
    return 0;
}

