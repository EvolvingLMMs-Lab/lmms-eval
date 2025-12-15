import math

from lmms_eval.tasks.gedit_bench.viescore import vie_prompts
from lmms_eval.tasks.gedit_bench.viescore.utils import mllm_output_to_dict


class VIEScore:
    def __init__(self, backbone="gpt4o", task="t2i", key_path=None) -> None:
        self.task = task
        self.backbone_name = backbone

        if self.task not in ["t2i", "tie", "t2v"]:
            raise ValueError("task must be either 't2i' or 'tie'")

        if self.backbone_name == "gpt4o":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.openai import GPT4o

            self.model = GPT4o(key_path, model_name="gpt-4.1")
        elif self.backbone_name == "gpt4v":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.openai import GPT4v

            self.model = GPT4v(key_path)
        elif self.backbone_name == "gemini":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.gemini import Gemini

            self.model = Gemini()
        elif self.backbone_name == "idefics2":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.idefics2_eval import (
                Idefics2,
            )

            self.model = Idefics2()
        elif self.backbone_name == "mantis":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.mantis_idefics2_eval import (
                Mantis,
            )

            self.model = Mantis()
        elif self.backbone_name == "minicpmv":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.minicpmv_eval import (
                MiniCPMV,
            )

            self.model = MiniCPMV()
        elif self.backbone_name == "qwen25vl":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.qwen25vl_eval import (
                Qwen25VL,
            )

            self.model = Qwen25VL()
        # vLLM-based backends (remote API)
        elif self.backbone_name == "vllm_qwen" or self.backbone_name == "vllm_qwen25vl":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.vllm_qwen_eval import (
                VLLMQwen25VL,
            )

            self.model = VLLMQwen25VL()
        elif self.backbone_name == "vllm_qwen3vl":
            from lmms_eval.tasks.gedit_bench.viescore.mllm_tools.vllm_qwen_eval import (
                VLLMQwen3VL,
            )

            self.model = VLLMQwen3VL()
        else:
            raise NotImplementedError(f"backbone '{backbone}' not supported. " f"Available: gpt4o, gpt4v, gemini, idefics2, mantis, minicpmv, qwen25vl, " f"vllm_qwen, vllm_qwen25vl, vllm_qwen3vl")
        self.context = vie_prompts._context_no_delimit
        if self.task == "t2i":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_image_gen_rule, vie_prompts._prompts_0shot_t2i_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "tie":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "t2v":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_video_gen_rule, vie_prompts._prompts_0shot_t2v_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_t2v_rule_PQ])

    def evaluate(self, image_prompts, text_prompt, extract_overall_score_only=False, extract_all_score=True, echo_output=False):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        if self.backbone_name in ["gpt4o", "gpt4v"]:
            self.model.use_encode = False if isinstance(image_prompts[0], str) else True
            # print("Using encode:", self.model.use_encode)
        if self.task == "t2i":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        elif self.task == "tie":
            _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt)
        elif self.task == "t2v":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        SC_prompt_final = self.model.prepare_prompt(image_prompts, _SC_prompt)
        if self.task == "tie":
            PQ_prompt_final = self.model.prepare_prompt(image_prompts[-1], self.PQ_prompt)
        else:
            PQ_prompt_final = self.model.prepare_prompt(image_prompts, self.PQ_prompt)

        results_dict = {}

        SC_dict = False
        PQ_dict = False
        tries = 0
        max_tries = 1
        while SC_dict is False or PQ_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_SC = self.model.get_parsed_output(SC_prompt_final)
            result_PQ = self.model.get_parsed_output(PQ_prompt_final)
            SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)
            PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=guess_if_cannot_parse)

        if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
            print("rate_limit_exceeded")
            raise ValueError("rate_limit_exceeded")
        results_dict["SC"] = SC_dict
        results_dict["PQ"] = PQ_dict
        if echo_output:
            print("results_dict", results_dict)
        if extract_all_score:
            SC_score = min(results_dict["SC"]["score"])
            PQ_score = min(results_dict["PQ"]["score"])
            O_score = math.sqrt(SC_score * PQ_score)
            return [SC_score, PQ_score, O_score]
        if extract_overall_score_only:
            SC_scores = results_dict["SC"]["score"]
            PQ_scores = results_dict["PQ"]["score"]
            O_score = math.sqrt(min(SC_scores) * min(PQ_scores))
            return O_score
        return results_dict


if __name__ == "__main__":
    model = VIEScore(backbone="gemini", task="t2i")
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/GenAI-Arena-Bench", "image_generation")
    dataset = dataset["test"]
    print("Now running the VIEScore model")
    for idx in range(5):
        left_image = dataset["left_image"][idx]
        right_image = dataset["right_image"][idx]
        prompt = dataset["prompt"][idx]
        print(model.evaluate(left_image, prompt, extract_all_score=True))
        print(model.evaluate(right_image, prompt, extract_all_score=True))
