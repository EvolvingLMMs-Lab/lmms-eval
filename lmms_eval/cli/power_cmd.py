"""lmms-eval power — statistical power analysis for benchmark planning."""

from __future__ import annotations

import argparse


def add_power_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "power",
        help="Run power analysis to plan benchmark sample sizes",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--effect-size", type=float, default=0.03, help="Minimum effect size to detect (default: 0.03 = 3%%)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    p.add_argument("--power", type=float, default=0.80, dest="stat_power", help="Desired statistical power (default: 0.80)")
    p.add_argument("--correlation", type=float, default=0.5, help="Expected correlation between paired samples (default: 0.5)")
    p.add_argument("--std-a", type=float, default=None, help="Std deviation of model A scores")
    p.add_argument("--std-b", type=float, default=None, help="Std deviation of model B scores")
    p.add_argument("--tasks", type=str, default=None, help="Comma-separated task names for per-task analysis")
    p.add_argument("--include-path", type=str, default=None, help="Additional task path")
    p.add_argument("--verbosity", type=str, default="WARNING", help="Logging verbosity")
    p.set_defaults(func=run_power)


def run_power(args: argparse.Namespace) -> None:
    import lmms_eval.tasks
    from lmms_eval.api.metrics import power_analysis
    from lmms_eval.tasks import TaskManager

    task_sizes: dict[str, int] = {}
    if args.tasks:
        task_manager = TaskManager(args.verbosity, include_path=args.include_path)
        task_names = task_manager.match_tasks(args.tasks.split(","))
        for task_name in task_names:
            task_dict = lmms_eval.tasks.get_task_dict([task_name], task_manager)
            for name, task_obj in task_dict.items():
                if hasattr(task_obj, "eval_docs"):
                    task_sizes[name] = len(task_obj.eval_docs)

    result = power_analysis(
        effect_size=args.effect_size,
        std_a=args.std_a,
        std_b=args.std_b,
        alpha=args.alpha,
        power=args.stat_power,
        correlation=args.correlation,
    )

    print("\n" + "=" * 60)
    print("POWER ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Effect size (delta):     {args.effect_size:.1%}")
    print(f"  Std (model A):           {result['std_a']}")
    print(f"  Std (model B):           {result['std_b']}")
    print(f"  Significance level (a):  {args.alpha}")
    print(f"  Desired power (1-b):     {args.stat_power}")
    print(f"  Correlation (p):         {args.correlation}")
    print(f"\nResult:")
    print(f"  Minimum sample size:     n = {result['min_n']}")
    print(f"\nInterpretation:")
    print(f"  To detect a {args.effect_size:.1%} difference with {args.stat_power:.0%} power,")
    print(f"  you need at least {result['min_n']} questions in your benchmark.")

    if task_sizes:
        print(f"\n" + "-" * 60)
        print("TASK ANALYSIS")
        print("-" * 60)
        for task_name, n_samples in task_sizes.items():
            task_result = power_analysis(
                effect_size=args.effect_size,
                std_a=args.std_a,
                std_b=args.std_b,
                alpha=args.alpha,
                power=args.stat_power,
                correlation=args.correlation,
                current_n=n_samples,
            )
            status = "OK Sufficient" if n_samples >= result["min_n"] else "X Insufficient"
            print(f"\n  {task_name}:")
            print(f"    Sample size:         n = {n_samples}")
            print(f"    Current power:       {task_result['current_power']:.1%}")
            print(f"    Min detectable d:    {task_result['min_detectable_effect']:.1%}")
            print(f"    Status:              {status}")

    print("\n" + "=" * 60 + "\n")
