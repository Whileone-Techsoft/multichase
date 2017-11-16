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

typedef unsigned atomic_t;

typedef union {
        struct {
                atomic_t count;
                int cpu;
		int counter;
        } x;
        char pad[AVOID_FALSE_SHARING];
} per_thread_t;

typedef union {
        struct {
                atomic_t count;
        } x[NUM_COUNTERS];
        char pad[CACHELINE_SIZE];
} global_t;

//Multiple counters to test performance under false sharing conditions.
global_t global_counter[COUNT_SWEEP_MAX];

static volatile int relaxed;
static volatile int count_sweep=0;
static volatile int sweep_active=1;


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
	int cid=args->x.counter;

        wait_for_startup();

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
					__sync_fetch_and_add(&args->x.count, 50*i);
			}

			if (delay_mask & (1u<<args->x.cpu)) {
					sleep(1);
			}
			while (relaxed) {
					for (i=0; i<inc_count ; i++) {
						x50(__sync_fetch_and_add(p, 1); cpu_relax(););
					}
					__sync_fetch_and_add(&args->x.count, 50*i);
			}
		}
        return NULL;
}

int main(int argc, char **argv)
{
        int c,sweep_count=1;
	size_t max_samples=6;
	int sample_interval=500000;
	int verbosity=0;
	static double spacer=0.;

        delay_mask = 0;
        while ((c = getopt(argc, argv, "d:s:n:t:v:N:r:")) != -1) {
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
                                "the startup.\n", argv[0]);
                exit(1);
        }

        setvbuf(stdout, NULL, _IONBF, BUFSIZ);

        // find the active cpus
        cpu_set_t cpus;
        if (sched_getaffinity(getpid(), sizeof(cpus), &cpus)) {
                perror("sched_getaffinity");
                exit(1);
        }

        // could do this more efficiently, but whatever
        size_t nr_threads = 0;
        int i;
        for (i = 0; i < CPU_SETSIZE; ++i) {
                if (CPU_ISSET(i, &cpus)) {
                        ++nr_threads;
                }
        }

        per_thread_t *thread_args = calloc(nr_threads, sizeof(*thread_args));
        nr_to_startup = nr_threads + 1;
        size_t u;
        i = 0;
        for (u = 0; u < nr_threads; ++u) {
                while (!CPU_ISSET(i, &cpus)) {
                        ++i;
                }
                thread_args[u].x.cpu = i;
                thread_args[u].x.counter = (int)((double)u * spacer) % NUM_COUNTERS;
                ++i;
                thread_args[u].x.count = 0;
                pthread_t dummy;
                if (pthread_create(&dummy, NULL, worker, &thread_args[u])) {
                        perror("pthread_create");
                        exit(1);
                }
        }

        wait_for_startup();

        atomic_t *samples = calloc(nr_threads, sizeof(*samples));

        printf("results are avg latency per locked increment in ns, one column per thread\n");
        printf("cpu,");
        for (u = 0; u < nr_threads; ++u) {
                printf("%u,", thread_args[u].x.cpu);
        }
        printf("avg,stdev,min,max\n");
	char msg[256];
	stat_t stats[2][COUNT_SWEEP_MAX];
	stat_t global_stats[2];
	sprintf(msg,"Unrelaxed summary across %d global counts [latency avg in ns, bw in ops/mSec]",COUNT_SWEEP_MAX);
	stat_init(&global_stats[0],msg);
	sprintf(msg,"Relaxed summary across %d global counts [latency avg in ns, bw in ops/mSec]",COUNT_SWEEP_MAX);
	stat_init(&global_stats[1],msg);
	if (sweep_count > COUNT_SWEEP_MAX)
		sweep_count = COUNT_SWEEP_MAX;
        for (count_sweep = 0; count_sweep < sweep_count; ++count_sweep) {
		for (relaxed = 0; relaxed < 2; ++relaxed) {
			sprintf(msg,"Global counter %d %s [bw in ops/mSec per thread, latency in ns]",count_sweep,relaxed ? "relaxed:" : "unrelaxed:");
			printf("%s\n",msg);
			stat_init(&stats[relaxed][count_sweep],msg);

			uint64_t last_stamp = now_nsec();
			size_t sample_nr;
			double bw=0.;
			for (sample_nr = 0; sample_nr < max_samples; ++sample_nr) {
				usleep(sample_interval);
				for (u = 0; u < nr_threads; ++u) {
						samples[u] = __sync_lock_test_and_set(&thread_args[u].x.count, 0);
				}
				uint64_t stamp = now_nsec();
				int64_t time_delta = stamp - last_stamp;
				last_stamp = stamp;

				// throw away the first sample to avoid race issues at startup / mode switch
				if (sample_nr == 0) continue;

				double sum = 0.;
				double sum_squared = 0.;
				double max=0.0,min=1e100;
				for (u = 0; u < nr_threads; ++u) {
						double s = time_delta / (double)samples[u];
						if (min > s) min=s;
						if (max < s) max=s;
						//Get BW stats in ops per msec instead of per ns.
						double bwt=(double)samples[u] * 1000000.0 / (double)time_delta;
						bw += bwt;
						printf(",%.1f", s);
						sum += s;
						sum_squared += s*s;
						stat_add(&stats[relaxed][count_sweep],s,bwt);
				}
				printf(",%.1f,%.1f,%.1f,%.1f\n",
						sum / nr_threads,
						sqrt((sum_squared - sum*sum/nr_threads)/(nr_threads-1)),
						min,max);
			}
			stat_update(&stats[relaxed][count_sweep]);
			stat_add(&global_stats[relaxed],
				stats[relaxed][count_sweep].latency.avg,
				bw/(double)sample_nr);
		}
	}
	sweep_active=0;
	if (verbosity > 0) { 
		stat_print(&stats[0][0]);
		stat_print(&stats[1][0]);
		if (verbosity > 1) {
			for (count_sweep = 1; count_sweep < sweep_count; ++count_sweep) {
				stat_print(&stats[0][count_sweep]);
			}
		}
		if ((sweep_count > 1) && (verbosity > 0) ) {
			stat_update(&global_stats[0]);
			stat_update(&global_stats[1]);
			stat_print(&global_stats[0]);
			stat_print(&global_stats[1]);
		}
	}
        return 0;
}

