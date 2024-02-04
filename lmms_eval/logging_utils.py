# Code mostly from: https://github.com/EleutherAI/lm-evaluation-harness/pull/1339, credit to: https://github.com/ayulockin
import copy
import logging
import re
import os
import json
import glob
import pandas as pd
from datetime import datetime

from packaging.version import Version

from lmms_eval import utils
import tenacity

logger = logging.getLogger(__name__)

IS_WANDB_AVAILABLE = False


try:
    import wandb

    assert Version(wandb.__version__) >= Version("0.16.0")
    # if Version(wandb.__version__) < Version("0.16.0"):
    #     wandb.require("report-editing:v0")

    IS_WANDB_AVAILABLE = True
except Exception as e:
    logger.warning("To use the wandb reporting functionality please install wandb>=0.16.0.\n" "To install the latest version of wandb run `pip install wandb --upgrade`\n" f"{e}")
    IS_WANDB_AVAILABLE = False


def remove_none_pattern(input_string):
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


class WandbLogger:
    def __init__(self, args):
        self.wandb_args = utils.simple_parse_args_string(args.wandb_args)
        self.args = args
        self.all_args_dict = vars(args)
        try:
            self.init_run()
        except Exception as e:
            logger.warning(f"Failed to initialize W&B run: {e}")
            os.environ["WANDB_MODE"] = "offline"
            self.init_run()

    @tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(5))
    def init_run(self):
        if "name" not in self.wandb_args:
            if "config" in self.all_args_dict and self.all_args_dict["config"] != "":
                self.wandb_args["name"] = self.all_args_dict["config"].split("/")[-1].split(".")[0]
            else:
                task_names = self.args.tasks.replace(",", "/")
                self.wandb_args["name"] = f"{self.args.model}_{task_names}_{self.args.log_samples_suffix}"
                if self.args.num_fewshot:
                    self.wandb_args["name"] += f"_{self.args.num_fewshot}shot"
        if "project" not in self.wandb_args:
            self.wandb_args["project"] = "lmms-eval"
        # initialize a W&B run
        self.run = wandb.init(**self.wandb_args)

        # call wr inside the init_run method to avoid multiple times logging
        import wandb.apis.reports as wr

        self.wr = wr

    def log_eval_result(self, results):
        # Log configs to wandb
        configs = self.get_config(results)
        self.run.config.update(configs)
        self.results = results

        wandb_summary, self.wandb_results = self.sanitize_results_dict()
        # update wandb.run.summary with items that were removed
        self.run.summary.update(wandb_summary)
        # Log the evaluation metrics to wandb
        self.run.log(self.wandb_results)
        # Log the evaluation metrics as Table
        self.get_eval_wandb_table()

    def get_eval_wandb_table(self):
        columns = ["Model Args", "Task", "Version", "Filter", "num_fewshot", "Metric", "Value", "Stderr"]
        table = wandb.Table(columns=columns)
        results = copy.deepcopy(self.results)
        model_args = results.get("config").get("model_args")
        for k, dic in results.get("results").items():
            version = results.get("versions").get(k)
            n = results.get("n-shot").get(k)

            if "alias" in dic:
                k = dic.pop("alias")

            for (mf), v in dic.items():
                m, _, f = mf.partition(",")
                if m.endswith("_stderr"):
                    continue

                if m + "_stderr" + "," + f in dic:
                    se = dic[m + "_stderr" + "," + f]
                    if se != "N/A":
                        se = "%.4f" % se
                    table.add_data(*[model_args, k, version, f, n, m, v, se])
                else:
                    table.add_data(*[model_args, k, version, f, n, m, v, ""])

        # log the table to W&B
        self.run.log({f"evaluation/eval_results": table})

    def generate_dataset(self, data, config):
        """Generate a Zeno dataset from evaluation data.

        Args:
            data: The data to generate a dataset for.
            config: The configuration of the task.

        Returns:
            pd.Dataframe: A dataframe that is ready to be uploaded to Zeno.
        """
        ids = [x["doc_id"] for x in data]
        labels = [x["target"] for x in data]
        instance = [""] * len(ids)

        metrics_list = config["metric_list"]
        metrics = {}
        for metric in metrics_list:
            metric = metric.get("metric")
            metrics[metric] = [x[metric] for x in data]

        if config["output_type"] == "loglikelihood":
            instance = [x["arguments"][0][0] for x in data]
            labels = [x["arguments"][0][1] for x in data]
        elif config["output_type"] == "multiple_choice":
            instance = [x["arguments"][0][0] + "\n\n" + "\n".join([f"- {y[1]}" for y in x["arguments"]]) for x in data]
        elif config["output_type"] == "loglikelihood_rolling":
            instance = [x["arguments"][0][0] for x in data]
        elif config["output_type"] == "generate_until":
            instance = [x["arguments"][0][0] for x in data]

        df_data = {
            "id": ids,
            "data": instance,
            "input_len": [len(x) for x in instance],
            "labels": labels,
            "output_type": config["output_type"],
        }
        df_data.update(metrics)

        return pd.DataFrame(df_data)

    def log_eval_samples(self, samples):
        task_names = list(self.results.get("results", {}).keys())
        for task_name in task_names:
            eval_preds = samples[task_name]
            df = self.generate_dataset(eval_preds, self.task_configs.get(task_name))
            self.run.log({f"{task_name}_eval_results": df})

    def get_config(self, results):
        task_configs = results.get("configs", {})
        cli_configs = results.get("config", {})
        configs = {
            "task_configs": task_configs,
            "cli_configs": cli_configs,
        }

        return configs

    def sanitize_results_dict(self):
        """
        Remove string valued keys from the results dict as they don't render in the workspace.
        Log these key-value pairs to wandb.summary.
        """
        _results = copy.deepcopy(self.results.get("results", dict()))

        task_names = list(self.results.get("results", {}).keys())
        # Remove None from the metric string name
        tmp_results = copy.deepcopy(_results)
        for task_name in task_names:
            task_result = tmp_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if removed:
                    _results[task_name][_metric_name] = metric_value
                    _results[task_name].pop(metric_name)

        # remove string valued keys from the results dict
        wandb_summary = {}
        for task in task_names:
            task_result = _results.get(task, dict())
            for metric_name, metric_value in task_result.items():
                if isinstance(metric_value, str):
                    wandb_summary[f"{task}/{metric_name}"] = metric_value

        for summary_metric, summary_value in wandb_summary.items():
            _task, _summary_metric = summary_metric.split("/")
            _results[_task].pop(_summary_metric)

        tmp_results = copy.deepcopy(_results)
        for task_name, task_results in tmp_results.items():
            for metric_name, metric_value in task_results.items():
                _results[f"{task_name}/{metric_name}"] = metric_value
                _results[task_name].pop(metric_name)
        for task in task_names:
            _results.pop(task)

        return wandb_summary, _results

    def prepare_report_by_task(self, results):
        task_names = list(results.get("results", {}).keys())
        blocks = []
        for task_name in task_names:
            blocks.append(self.wr.H2(task_name))
            panels = []
            for metric_name, metric_value in results.items():
                if task_name in metric_name:
                    panels.append(
                        self.wr.ScalarChart(
                            title=f"{metric_name}",
                            metric=f"{metric_name}",
                            font_size="large",
                        )
                    )
            _results = {
                "results": {f"{task_name}": results.get("results").get(task_name)},
                "versions": {f"{task_name}": results.get("versions").get(task_name)},
                "n-shot": {f"{task_name}": results.get("n-shot").get(task_name)},
            }
            results_md = utils.make_table(_results)
            blocks.extend([self.wr.MarkdownBlock(results_md), self.wr.PanelGrid(panels=panels)])
            # blocks.extend([
            #     self.wr.WeaveBlockSummaryTable(
            #         project=self.run.project,
            #         entity=self.run.entity,
            #         table_name=f"{task_name}_eval_results",
            #     ),
            #     self.wr.PanelGrid(
            #         runsets=[
            #             self.wr.Runset(
            #                 project=self.run.project, entity=self.run.entity,
            #             ).set_filters_with_python_expr(f'Name == "{str(self.run.name)}"'),
            #         ]
            #     ),
            # ])

        return blocks

    def write_to_report(self, results):
        wandb_project = self.run.project
        wandb_entity = self.run.entity
        report = self.wr.Report(
            project=self.run.project,
            entity=self.run.entity,
            title=f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) xxx - Evaluation report",
            description=f"Evaluation run by: {self.run.entity} logged to {self.run.url}",
        )

        results_md = utils.make_table(results)
        task_blocks = self.prepare_report_by_task(self.wandb_results)

        blocks = (
            [
                self.wr.TableOfContents(),
                self.wr.H1("Complete Evaluation Results"),
                self.wr.WeaveBlockSummaryTable(
                    project=self.run.project,
                    entity=self.run.entity,
                    table_name=f"evaluation/eval_results",
                ),
                self.wr.PanelGrid(
                    runsets=[
                        self.wr.Runset(
                            project=self.run.project,
                            entity=self.run.entity,
                        ).set_filters_with_python_expr(f'Name == "{str(self.run.name)}"'),
                    ]
                ),
                self.wr.H1("Evaluation Results By Task"),
            ]
            + task_blocks
            + [
                self.wr.H1("Evaluation Config"),
                self.wr.CodeBlock(json.dumps(self.results["config"], indent=5).split("\n"), language="json"),
                # TODO: Add appendix
            ]
        )

        report.blocks = blocks
        report.save()
        wandb.termlog(f"📝 Check out the autogenerated report at: {report.url}")

    def finish(self):
        self.run.finish()
