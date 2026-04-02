from src.stats_tools import one_gram
from src.core.stats.common_runner import evaluate_single_run


def run_one_gram(mode: str = "single") -> None:
    use_wordpress = True
    dataset = "WordPress" if use_wordpress else "Nextcloud"

    assignment_mode = "true"
    assignment_idx = None

    output_path = f"results/statistic_one_gram{'_wordpress' if use_wordpress else '_nextcloud'}.csv"

    if mode == "single":
        config = {
            "dataset": dataset,
            "log_type": "syslog",
            "mode": "char",
            "metric": "js",
            "min_count": 1,
        }

        result = one_gram.run(config)

        evaluate_single_run(
            tool_name="one_gram",
            labels=result["labels"],
            pairwise_results=result["pairwise_results"],
            distance_name=config["metric"],
            distance_extractor=lambda item: item[config["metric"]],
            hyperparameters=config,
            assignment_mode=assignment_mode,
            assignment_idx=assignment_idx,
            plot=True,
            output_path=None,
        )

    elif mode == "sweep":
        log_types = ["audit", "syslog"] if use_wordpress else ["audit", "syslog", "nextcloud"]
        modes = ["word", "char"]
        metrics = ["l1", "js"]

        configs = one_gram.build_sweep_configs(
            dataset=dataset,
            log_types=log_types,
            modes=modes,
            metrics=metrics,
            min_count=1,
        )

        print(
            f"\nRunning one_gram sweep with: "
            f"dataset={dataset}, "
            f"assignment_mode={assignment_mode}, "
            f"assignment_idx={assignment_idx}, "
            f"output_path={output_path}"
        )

        for i, config in enumerate(configs, 1):
            print(
                f"\n[{i}/{len(configs)}] "
                f"log_type={config['log_type']} "
                f"mode={config['mode']} "
                f"metric={config['metric']}"
            )

            result = one_gram.run(config)

            evaluate_single_run(
                tool_name="one_gram",
                labels=result["labels"],
                pairwise_results=result["pairwise_results"],
                distance_name=config["metric"],
                distance_extractor=lambda item: item[config["metric"]],
                hyperparameters={
                    "log_type": config["log_type"],
                    "mode": config["mode"],
                    "min_count": config["min_count"],
                    "assignment_mode": assignment_mode,
                    "assignment_idx": assignment_idx,
                },
                assignment_mode=assignment_mode,
                assignment_idx=assignment_idx,
                output_path=output_path,
                plot=False,
            )

    else:
        raise ValueError("mode must be 'single' or 'sweep'")
    


if __name__ == "__main__":

    run_one_gram("single")
