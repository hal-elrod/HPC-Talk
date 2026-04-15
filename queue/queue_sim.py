#!/usr/bin/env python3
"""
Queue Simulation using SimPy
Simulates a single-arrival, multi-server (G/G/N) queue system.
Hal Elrod - helrod@gmail.com

Usage:
    python3 queue_sim.py [options]

Distribution modes:
    exponential (default): inter-arrival times drawn from Exp(mean),
        service times drawn from LogNormal(mean, std).
        -a / --arrival-mean   mean inter-arrival time  (default: 10)
        -s / --handle-mean    mean service/handle time (default: 10)
        --service-std         standard deviation for service time lognormal distribution (default: 2)

    uniform: inter-arrival and service times drawn from Uniform(low, high).
        --dist uniform --arrival LOW HIGH --service LOW HIGH

Examples:
    python3 queue_sim.py                              # Exp(10) arrivals, LogNormal(10, 2) service, 1 server
    python3 queue_sim.py -a 8 -s 12 --service-std 1.5 --queues 2
                                                     # Exp(8) arrivals, LogNormal(12, 1.5) service, 2 servers
    python3 queue_sim.py --dist uniform --arrival 2 8 --service 5 10 --queues 2
    python3 queue_sim.py --queues 4 --customers 500 --seed 42
    python3 queue_sim.py --compare                    # side-by-side 1/2/4 servers
"""

import simpy
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy import stats


@dataclass
class CustomerRecord:
    customer_id: int
    arrival_time: float
    service_start_time: float
    service_end_time: float
    server_id: int

    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue before service begins."""
        return self.service_start_time - self.arrival_time

    @property
    def service_time(self) -> float:
        """Time spent being actively served."""
        return self.service_end_time - self.service_start_time

    @property
    def total_time(self) -> float:
        """Total time in system (wait + service)."""
        return self.service_end_time - self.arrival_time


class QueueSimulation:
    """
    Discrete-event queue simulation using SimPy.

    Models a G/G/N queue: general inter-arrival times, general service times,
    N parallel servers fed by a single waiting line.

    Distribution options:
        "exponential"  — Exp(mean) inter-arrivals with LogNormal(mean, std) service.
        "uniform"      — Uniform(low, high).
    """

    def __init__(
        self,
        num_queues: int = 1,
        dist: str = "exponential",
        # exponential parameters
        arrival_mean: float = 10.0,
        service_mean: float = 10.0,
        service_std: float = 2.0,
        # uniform parameters
        arrival_low: float = 5.0,
        arrival_high: float = 15.0,
        service_low: float = 5.0,
        service_high: float = 15.0,
        num_customers: int = 200,
        seed: int = None,
    ):
        if num_queues not in (1, 2, 4):
            raise ValueError("num_queues must be 1, 2, or 4")
        if dist not in ("exponential", "uniform"):
            raise ValueError("dist must be 'exponential' or 'uniform'")
        if dist == "exponential":
            if arrival_mean <= 0 or service_mean <= 0 or service_std <= 0:
                raise ValueError("arrival_mean, service_mean, and service_std must be positive")
        else:
            if arrival_low >= arrival_high:
                raise ValueError("arrival_low must be less than arrival_high")
            if service_low >= service_high:
                raise ValueError("service_low must be less than service_high")

        self.num_queues = num_queues
        self.dist = dist
        self.arrival_mean = arrival_mean
        self.service_mean = service_mean
        self.service_std = service_std
        self.arrival_low = arrival_low
        self.arrival_high = arrival_high
        self.service_low = service_low
        self.service_high = service_high
        self.num_customers = num_customers
        self.seed = seed

        self.records: List[CustomerRecord] = []
        self._queue_length_samples: List[tuple] = []  # (time, length)

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def _dist_label(self, role: str) -> str:
        """Human-readable label for the configured distribution."""
        if self.dist == "exponential":
            if role == "arrival":
                return f"Exponential(mean={self.arrival_mean})"
            return f"LogNormal(mean={self.service_mean}, std={self.service_std})"
        else:
            low  = self.arrival_low  if role == "arrival" else self.service_low
            high = self.arrival_high if role == "arrival" else self.service_high
            return f"Uniform({low}, {high})"

    def _theoretical_means(self) -> Tuple[float, float]:
        """Return (mean_interarrival, mean_service) for traffic intensity calculation."""
        if self.dist == "exponential":
            return self.arrival_mean, self.service_mean
        else:
            return (
                (self.arrival_low + self.arrival_high) / 2,
                (self.service_low + self.service_high) / 2,
            )

    def _sample_interarrival(self) -> float:
        if self.dist == "exponential":
            return random.expovariate(1.0 / self.arrival_mean)
        return random.uniform(self.arrival_low, self.arrival_high)

    def _sample_service(self) -> float:
        if self.dist == "exponential":
            # LogNormal: convert desired mean/std to underlying normal parameters
            import math
            mu = self.service_mean
            sigma = self.service_std
            sigma2 = math.log(1 + (sigma / mu) ** 2)
            mu_ln = math.log(mu) - sigma2 / 2
            return random.lognormvariate(mu_ln, math.sqrt(sigma2))
        return random.uniform(self.service_low, self.service_high)

    def _arrival_process(self, env: simpy.Environment, servers: simpy.Resource):
        for i in range(self.num_customers):
            yield env.timeout(self._sample_interarrival())
            env.process(self._customer(env, servers, i))

    def _customer(self, env: simpy.Environment, servers: simpy.Resource, cid: int):
        arrival = env.now
        self._queue_length_samples.append((arrival, len(servers.queue)))

        with servers.request() as req:
            yield req
            service_start = env.now
            svc = self._sample_service()
            # Track which server slot is handling this customer
            server_id = list(servers.users).index(req) if req in servers.users else -1
            yield env.timeout(svc)
            service_end = env.now

        self.records.append(
            CustomerRecord(
                customer_id=cid,
                arrival_time=arrival,
                service_start_time=service_start,
                service_end_time=service_end,
                server_id=server_id,
            )
        )

    def run(self):
        """Run the simulation and collect statistics."""
        if self.seed is not None:
            random.seed(self.seed)

        self.records = []
        self._queue_length_samples = []

        env = simpy.Environment()
        servers = simpy.Resource(env, capacity=self.num_queues)
        env.process(self._arrival_process(env, servers))
        env.run()

        return self

    def summary(self) -> dict:
        """Return a dict of computed statistics."""
        if not self.records:
            return {}

        wait = np.array([r.wait_time for r in self.records])
        svc = np.array([r.service_time for r in self.records])
        total = np.array([r.total_time for r in self.records])

        sim_duration = self.records[-1].service_end_time
        total_busy = svc.sum()
        utilization = total_busy / (sim_duration * self.num_queues)

        mean_interarrival, mean_service = self._theoretical_means()
        traffic_intensity = mean_service / (mean_interarrival * self.num_queues)

        avg_queue_len = (
            np.mean([q for _, q in self._queue_length_samples])
            if self._queue_length_samples
            else 0.0
        )

        # 95% confidence interval for mean wait time (undefined if all waits are zero)
        n = len(wait)
        sem = stats.sem(wait)
        if sem > 0:
            ci = stats.t.interval(0.95, df=n - 1, loc=wait.mean(), scale=sem)
        else:
            ci = (wait.mean(), wait.mean())

        return {
            "num_customers": len(self.records),
            "sim_duration": sim_duration,
            "wait": {
                "mean": wait.mean(),
                "median": np.median(wait),
                "std": wait.std(),
                "min": wait.min(),
                "max": wait.max(),
                "ci95": ci,
                "pct_nonzero": (wait > 1e-6).mean(),
            },
            "service": {
                "mean": svc.mean(),
                "median": np.median(svc),
                "std": svc.std(),
                "min": svc.min(),
                "max": svc.max(),
            },
            "total": {
                "mean": total.mean(),
                "median": np.median(total),
                "std": total.std(),
                "min": total.min(),
                "max": total.max(),
            },
            "utilization": utilization,
            "traffic_intensity": traffic_intensity,
            "avg_queue_length": avg_queue_len,
        }

    def print_report(self):
        """Print a formatted statistics report."""
        s = self.summary()
        if not s:
            print("No data — run the simulation first.")
            return

        W = s["wait"]
        SV = s["service"]
        T = s["total"]

        print()
        print("╔══ Queue Simulation Report")
        print("╠══ Configuration")
        print(f"║  Servers (parallel queues) : {self.num_queues}")
        print(f"║  Inter-arrival distribution: {self._dist_label('arrival')}")
        print(f"║  Service time distribution : {self._dist_label('service')}")
        print(f"║  Customers simulated       : {s['num_customers']}")
        print(f"║  Simulation duration       : {s['sim_duration']:.2f}")
        print("╠══ Wait Time  (time in queue before service)")
        print(f"║  Mean   : {W['mean']:.4f}    Std Dev : {W['std']:.4f}")
        print(f"║  Median : {W['median']:.4f}    Min     : {W['min']:.4f}")
        print(f"║  Max    : {W['max']:.4f}    % waited: {W['pct_nonzero']:.1%}")
        print(f"║  95% CI (mean) : [{W['ci95'][0]:.4f}, {W['ci95'][1]:.4f}]")
        print("╠══ Service Time  (time actively being served)")
        print(f"║  Mean   : {SV['mean']:.4f}    Std Dev : {SV['std']:.4f}")
        print(f"║  Median : {SV['median']:.4f}    Min     : {SV['min']:.4f}")
        print(f"║  Max    : {SV['max']:.4f}")
        print("╠══ Total Time in System  (wait + service)")
        print(f"║  Mean   : {T['mean']:.4f}    Std Dev : {T['std']:.4f}")
        print(f"║  Median : {T['median']:.4f}    Min     : {T['min']:.4f}")
        print(f"║  Max    : {T['max']:.4f}")
        print("╠══ System Metrics")
        print(f"║  Server utilization    : {s['utilization']:.1%}")
        print(f"║  Traffic intensity (ρ) : {s['traffic_intensity']:.4f}")
        print(f"║  Avg queue length (Lq) : {s['avg_queue_length']:.4f}")
        stability = "stable ✓" if s["traffic_intensity"] < 1.0 else "UNSTABLE ✗ (ρ ≥ 1)"
        print(f"║  Queue stability       : {stability}")
        print("╚")
        print()


def compare_configurations(
    dist: str = "exponential",
    arrival_mean: float = 10.0,
    service_mean: float = 10.0,
    service_std: float = 2.0,
    arrival_low: float = 5.0,
    arrival_high: float = 15.0,
    service_low: float = 5.0,
    service_high: float = 15.0,
    num_customers: int = 200,
    seed: int = 42,
):
    """Run simulations for 1, 2, and 4 servers and print a comparison table."""
    results = {}
    for n in (1, 2, 4):
        sim = QueueSimulation(
            num_queues=n,
            dist=dist,
            arrival_mean=arrival_mean,
            service_mean=service_mean,
            service_std=service_std,
            arrival_low=arrival_low,
            arrival_high=arrival_high,
            service_low=service_low,
            service_high=service_high,
            num_customers=num_customers,
            seed=seed,
        )
        sim.run()
        results[n] = sim.summary()

    print()
    print("╔══ Comparison: 1 vs 2 vs 4 Servers")
    print(f"║  {'Metric':<24}  {'1 server':>10}  {'2 servers':>10}  {'4 servers':>10}")
    print("╠" + "─" * 62)

    rows = [
        ("Avg wait time",        lambda s: s["wait"]["mean"],        ".2f"),
        ("Avg service time",     lambda s: s["service"]["mean"],     ".2f"),
        ("Avg total time",       lambda s: s["total"]["mean"],       ".2f"),
        ("Max wait time",        lambda s: s["wait"]["max"],         ".2f"),
        ("% customers waited",   lambda s: s["wait"]["pct_nonzero"], ".1%"),
        ("Server utilization",   lambda s: s["utilization"],         ".1%"),
        ("Traffic intensity ρ",  lambda s: s["traffic_intensity"],   ".4f"),
        ("Avg queue length Lq",  lambda s: s["avg_queue_length"],    ".4f"),
    ]

    for label, fn, fmt in rows:
        vals = [format(fn(results[n]), fmt) for n in (1, 2, 4)]
        print(f"║  {label:<24}  {vals[0]:>10}  {vals[1]:>10}  {vals[2]:>10}")

    print("╚")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Queue simulation (G/G/N) using SimPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--queues",
        type=int,
        choices=[1, 2, 4],
        default=1,
        metavar="{1,2,4}",
        help="Number of parallel servers/queues (default: 1)",
    )
    parser.add_argument(
        "--dist",
        choices=["exponential", "uniform"],
        default="exponential",
        help="Distribution mode (default: exponential = Exp arrivals + LogNormal service)",
    )
    # Exponential parameters
    parser.add_argument(
        "-a", "--arrival-mean",
        type=float,
        default=10.0,
        metavar="MEAN",
        dest="arrival_mean",
        help="Mean inter-arrival time for exponential distribution (default: 10)",
    )
    parser.add_argument(
        "-s", "--handle-mean",
        type=float,
        default=10.0,
        metavar="MEAN",
        dest="service_mean",
        help="Mean service/handle time for lognormal service distribution when --dist exponential (default: 10)",
    )
    parser.add_argument(
        "--service-std",
        type=float,
        default=2.0,
        metavar="STD",
        dest="service_std",
        help="Std dev for lognormal service distribution when --dist exponential (default: 2)",
    )
    # Uniform parameters
    parser.add_argument(
        "--arrival",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=[5.0, 15.0],
        help="Inter-arrival time range for uniform distribution (default: 5 15)",
    )
    parser.add_argument(
        "--service",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=[5.0, 15.0],
        help="Service time range for uniform distribution (default: 5 15)",
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=200,
        help="Number of customers to simulate (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run and compare 1, 2, and 4 server configurations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    arrival_low, arrival_high = args.arrival
    service_low, service_high = args.service

    if args.compare:
        compare_configurations(
            dist=args.dist,
            arrival_mean=args.arrival_mean,
            service_mean=args.service_mean,
            service_std=args.service_std,
            arrival_low=arrival_low,
            arrival_high=arrival_high,
            service_low=service_low,
            service_high=service_high,
            num_customers=args.customers,
            seed=args.seed if args.seed is not None else 42,
        )
    else:
        sim = QueueSimulation(
            num_queues=args.queues,
            dist=args.dist,
            arrival_mean=args.arrival_mean,
            service_mean=args.service_mean,
            service_std=args.service_std,
            arrival_low=arrival_low,
            arrival_high=arrival_high,
            service_low=service_low,
            service_high=service_high,
            num_customers=args.customers,
            seed=args.seed,
        )
        sim.run()
        sim.print_report()
