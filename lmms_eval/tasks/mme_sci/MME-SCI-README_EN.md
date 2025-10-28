**PAPER**: [MME-SCI](https://www.arxiv.org/abs/2508.13938)

##### How to Evaluate — **Usage of run_eval.sh:**

```
./run_eval.sh <TASK> <CKPT_PATH> <CONV_TEMPLATE> <MODEL_NAME>
```

This will provide the results. An example is shown below:

###### mme_sci:

```
cd lmms-eval
./run_eval.sh mme_sci PATH/TO/Qwen2___5-VL-3B-Instruct qwen_vl qwen2_5_vl_chat
```

This will output two files:

```
lmms-eval/logs/Qwen2___5-VL-3B-Instruct/20250928_152715_results.json
lmms-eval/logs/Qwen2___5-VL-3B-Instruct/20250928_152715_samples_mme_sci.jsonl
```

`0250928_152715` is a timestamp from lmms-eval.

###### **mme_sci_image:**

```
cd lmms-eval
./run_eval.sh mme_sci_image PATH/TO/Qwen2___5-VL-3B-Instruct qwen_vl qwen2_5_vl_chat
```

**------------------------------------------------------------------------------------------------------------------------------------------**

#### Enabling SgLang for Local Evaluation

**Install dependencies in a new environment to avoid conflicts:**

```
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.6.post4"
```

Reference: [Install SGLang — SGLang Framework](https://docs.sglang.com.cn/start/install.html)

##### **Explanation of run_judge_sglang.py:**

###### Why do we need _from_ openai _import_ OpenAI?

SgLang’s HTTP interface is compatible with the OpenAI style. The `api_key = os.environ.get("OPENAI_API_KEY", "sk-local")` is custom and does not require a real token. `client = OpenAI(api_key=api_key, base_url=f"http://{HOST}:{PORT}/v1")` is just an HTTP wrapper.

```
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from lmms_eval.llm_judge.launcher.sglang import SGLangLauncher
# This part uses the SGLangLauncher from "lmms-eval\lmms_eval\llm_judge\launcher\sglang.py", remember to modify it to your own path
```

Also, remember to modify `INPUT_FILE` and `OUTPUT_FILE` to your own paths.

```
python run_judge_sglang.py
```

This will directly give you the results. The terminal logs will scroll quickly because SGLang internally prints a lot of `[INFO]` and prefill logs, but you can ignore them.

The judge output can be found at: lmms-eval/logs/judge_output

The final result printed in the logs will look like this:

zh:

```
Judging samples: 100%|███████████████████████████| 1019/1019 [02:14<00:00,  7.55it/s]
[INFO] Judging complete.
[INFO] Total valid samples: 1019
[INFO] Correct: 260
[INFO] Accuracy: 25.52%
```

img:

```
Judging samples: 100%|███████████████████████████| 1019/1019 [02:40<00:00,  6.35it/s]
[INFO] Judging complete.
[INFO] Total valid samples: 1019
[INFO] Correct: 334
[INFO] Accuracy: 32.78%
```
