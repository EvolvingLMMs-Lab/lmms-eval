# seephys_evals.py
import ast
import os
import re
import time
from typing import Dict, Any, Optional
import yaml
from loguru import logger as eval_logger
from openai import OpenAI
from sympy.parsing.latex import parse_latex
from sympy import latex, Eq, simplify
from pathlib import Path
import json

FAIL_MSG = "Failed to obtain answer via API."

def load_seephys_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "seephys.yaml"
    eval_logger.info(f"Loading SeePhys config from: {config_path}")

    if not config_path.exists():
        eval_logger.error(f"Config file not found at: {config_path}")
        return {"metadata": {}}

    with open(config_path, "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for line in raw_data:
            if "!function" not in line:
                safe_data.append(line)

    config_data = "".join(safe_data)
    if not config_data.strip():
        eval_logger.error(f"Config file {config_path} is empty or only contains !function tags.")
        return {"metadata": {}}

    try:
        config = yaml.safe_load(config_data)
        if "metadata" not in config:
            eval_logger.warning(f"'metadata' block not found in {config_path}. Using empty metadata.")
            config["metadata"] = {}
        return config
    except yaml.YAMLError as e:
        eval_logger.error(f"Error parsing YAML config {config_path}: {e}")
        return {"metadata": {}}

config = load_seephys_config()

if "metadata" not in config:
    raise ValueError("Could not load metadata from seephys.yaml. Check file content and permissions.")


class SeephysEvaluator:
    def __init__(self):
        self.juder_model = None
        self.headers = None
        self.client = None

        if not config.get("metadata", {}).get("quick_extract", False):
            self.juder_model = config["metadata"].get("eval_model_name")
            API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            API_KEY = os.getenv("OPENAI_API_KEY", "")
            if not API_KEY:
                eval_logger.error("OPENAI_API_KEY not found. Please set the environment variable: `export OPENAI_API_KEY=$Your_KEY`")
                raise ValueError("OPENAI_API_KEY is required for LLM-as-a-judge evaluation.")
            
            self.client = OpenAI(api_key=API_KEY, base_url=API_URL)
            eval_logger.debug(f"Initialized SeephysEvaluator judger client for model {self.juder_model}")

    def _safe_to_dict(self, resp) -> Dict[str, Any]:
        """
        Convert response to plain dict for robust inspection.
        """
        try:
            if hasattr(resp, "to_dict"):
                return resp.to_dict()
            if isinstance(resp, dict):
                return resp
            # fallback: json dumps/loads
            return json.loads(json.dumps(resp))
        except Exception as e:
            eval_logger.debug(f"_safe_to_dict error: {e}")
            return {}

    def _extract_content_from_response(self, resp_dict: Dict[str, Any]) -> str:
        """
        Robustly extract visible content from various possible response formats.
        """
        try:
            choices = resp_dict.get("choices", [])
            if choices and isinstance(choices, list):
                c0 = choices[0]
               
                if isinstance(c0, dict):
                    msg = c0.get("message") or c0.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if content:
                            return content.strip()
                
                text = c0.get("text")
                if text:
                    return text.strip()
        except Exception as e:
            eval_logger.debug(f"extract content try1 error: {e}")

        for key in ("output_text", "generated_text", "content"):
            if key in resp_dict and resp_dict[key]:
                try:
                    return str(resp_dict[key]).strip()
                except Exception:
                    pass

        return ""

    def judger_generate(self, prompt: str, temperature: int = 1, max_completion_tokens: int = 4096, n: int = 1, patience: int = 3, sleep_time: int = 0) -> str:
        """
        Call the judger LLM and try to robustly return visible content.
        If the first call returns no visible content but shows evidence of internal reasoning tokens,
        attempt a single retry with an increased `max_completion_tokens`.
        """
        if not self.client:
            eval_logger.error("LLM judger client not initialized. Ensure quick_extract=False and API key is set.")
            return FAIL_MSG

        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.juder_model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_completion_tokens is None:
            max_completion_tokens = 4096
        payload["max_completion_tokens"] = int(max_completion_tokens)

        attempt = 0
        while attempt < patience:
            attempt += 1
            try:
                resp = self.client.chat.completions.create(**payload)
                resp_dict = self._safe_to_dict(resp)
                eval_logger.debug(f"Judger raw response (attempt {attempt}): {json.dumps(resp_dict, indent=2)[:8000]}")
                content = self._extract_content_from_response(resp_dict)

                
                usage = resp_dict.get("usage", {}) or {}
                finish_reason = ""
                try:
                    choices = resp_dict.get("choices", [])
                    if choices and isinstance(choices, list):
                        finish_reason = choices[0].get("finish_reason", "") or ""
                except Exception:
                    finish_reason = ""

                if content:
                    return content.strip()

                reasoning_tokens = 0
                try:
                    comp_details = usage.get("completion_tokens_details") or usage.get("completion_tokens") or {}
                    if isinstance(comp_details, dict):
                        reasoning_tokens = int(comp_details.get("reasoning_tokens", 0) or 0)
                    reasoning_tokens = int(usage.get("reasoning_tokens", reasoning_tokens) or reasoning_tokens)
                except Exception:
                    reasoning_tokens = 0

                if (not content) and (finish_reason == "length" or reasoning_tokens > 0):
                    eval_logger.warning(f"Judger produced no visible content on attempt {attempt}. finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}. Retrying with larger max_completion_tokens.")
                    payload["max_completion_tokens"] = int(max(4096, payload.get("max_completion_tokens", 0) * 2))
                    time.sleep(sleep_time)
                    continue

                if not content:
                
                    textual = json.dumps(resp_dict, ensure_ascii=False)[:8000]
                    return textual

            except Exception as e:
                eval_logger.warning(f"LLM judger API call failed (attempt {attempt}): {e}")
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

        return FAIL_MSG

    def get_ICE_scoring(self) -> str:
        return r"""
You are a physics professor, please determine if the Standard answer and Model Answer are equivalent. Note that the significant figures in the answer must meet the requirements. Your judgment should be 0 (non-equivalent) or 1 (equivalent).

[Question]: A force of 20 N acts on an object of mass 5 kg. What is the acceleration of the object?
[Standard Answer]: 4 m/s²
[Model Answer] : 4
Judgement: 1

[Question]: A projectile is launched at an angle $\\theta$ with initial velocity $v_0$. What is its time of flight before returning to the same height, assuming negligible air resistance and gravitational acceleration $g$?
[Standard Answer]: $$ t = \\frac{{2 v_0 \\sin(\\theta)}}{{g}} $$
[Model Answer] : Extracted Answer: $$ t = \\frac{{2 v_0 \\cos(\\frac{\\pi}{2} - \\theta)}}{{g}} $$
Judgement: 1

[Question]: The position of a particle is given by $x(t) = 3t^2 - 2t + 5$ meters. What is its instantaneous velocity at $t=2$ seconds?
[Standard Answer]: 10 m/s
[Model Answer] : Velocity $v(t) = dx/dt = 6t - 2$. At $t=2s$, $v(2) = 6(2) - 2 = 12 - 2 = 10$. So the velocity is 10 m/s.
Judgement: 1

[Question]: A car travels North at 20 m/s. It then turns and travels East at 20 m/s. What is the magnitude of its change in velocity?
[Standard Answer]: Approximately 28.3 m/s
[Model Answer] : The change in velocity is 0 m/s because the speed is the same.
Judgement: 0

[Question]: An object is thrown horizontally from a height of 20m with an initial speed of 10 m/s. Calculate: (a) the time it takes to hit the ground ($t_g$), and (b) the horizontal distance ($d_x$) it travels before hitting the ground. (Use g = 10 m/s²)
[Standard Answer]: (a) $t_g = 2$ s, (b) $d_x = 20$ m
[Model Answer] : (a) The time to hit the ground $t_g$ is 2 s. (b) The horizontal distance $d_x$ is 10 m.
Judgement: 0

[Question]: An engine performs $1.2 \\times 10^5$ J of work in 2 minutes. What is its average power output in watts?
[Standard Answer]: 1 kW
[Model Answer] : Power = Work / Time = $1.2 \\times 10^5$ J / (2 min * 60 s/min) = $1.2 \\times 10^5$ J / 120 s = 1000 W.
Judgement: 1

[Question]: A resistor has a voltage of 10V across it and a current of 2A flowing through it. What is its resistance and power dissipation?
[Standard Answer]: Resistance R = 5 Ohms , Power P = 20 Watts.
[Model Answer] : The resistance is $R = V/I = 10V / 2A = 5 \Omega$. The power dissipated is $P = VI = 10V \\times 2A = 20W$.
Judgement: 1

[Question]: The displacement of an object in Simple Harmonic Motion (SHM) is given by $x(t) = A \sin(\omega t)$. Determine the equation for its acceleration, $a(t)$.
[Standard Answer]: $$ a(t) = -A\omega^2 \sin(\omega t) $$
[Model Answer] : The acceleration is the second derivative of displacement. $v(t) = A\omega \cos(\omega t)$. $a(t) = A\omega^2 \cos\left(\omega t + \\frac{\pi}{2}\\right)$.
Judgement: 1

[Question]: 给出相对论性粒子总能量 $E$ 的速度展开式（到 $v^4/c^4$ 项）。
[Standard Answer]: $E = mc^2 \left(1 + \frac{v^2}{2c^2} + \frac{3v^4}{8c^4} + \mathcal{O}(v^6/c^6)\right)$
[Model Answer]: $E = \gamma m c^2 = \frac{mc^2}{\sqrt{1 - v^2/c^2}} \approx mc^2 + \frac{1}{2}mv^2 + \frac{3}{8} \frac{mv^4}{c^2}$
Judgement: 1

[Question]: 计算粒子能量 $E$ 穿过势垒 $V_0$ ($E < V_0$) 的透射系数 $T$。
[Standard Answer]: $\ln T \approx \ln 16 + \ln\left(\frac{E}{V_0}\right) + \ln\left(1 - \frac{E}{V_0}\right) - \frac{2d}{\hbar} \sqrt{2m(V_0 - E)}$
[Model Answer]: $T \approx 16 \frac{E}{V_0} \left(1 - \frac{E}{V_0}\right) e^{-2d\sqrt{2m(V_0 - E)}/\hbar}$
Judgement: 1

[Question]: The position of a particle is given by $x(t) = (2t^3 - 3t)$ meters. What is its acceleration at $t=1$ second? The final answer should retain 3 significant figures.
[Standard Answer]: 12.0 m/s²
[Model Answer] : $v(t) = 6t^2 - 3$. $a(t) = 12.1t$. At $t=1s$, $a(1) = 12.1 \\text{ m/s}^2$.
Judgement: 0
---
Now please provide your judgement (0 or 1), DONNOT output explanation:
"""

    def build_seephys_scoring_prompt(self, line: dict, pred: str) -> str:
        query = line.get("question", "")
        gt = line.get("answer", "")
        
        full_prompt = self.get_ICE_scoring().strip() + \
                      f"\n[Question]: {query}\n[Standard Answer]: {gt}\n[Model Answer]: {pred}\nJudgement: "
        return full_prompt

    def _extract_answer_by_rule(self, line: dict) -> str:
        """
        Robust extraction:
        1) Try <answer>...<\answer> (case-insensitive, flexible whitespace)
        2) If not found, try to extract last non-empty line that looks numeric or latex
        3) Try to extract last math expression pattern
        4) Fallback to returning full response
        """
        response = line.get('prediction', '') or ''
        response = str(response).strip()
        if not response:
            return ""

        # 1) try <answer> tag
        try:
            pattern = r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>"
            m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if m:
                ans = m.group(1).strip()
                # strip wrapping $ signs
                ans = re.sub(r'^\$+|\$+$', '', ans).strip()
                return ans
        except Exception:
            pass

        # 2) try after </think>
        try:
            m = re.search(r"</\s*think\s*>\s*(.*)", response, re.DOTALL | re.IGNORECASE)
            if m:
                tail = m.group(1).strip()
                # If tail contains <answer> extract that first (redundant but safe)
                m2 = re.search(pattern, tail, re.DOTALL | re.IGNORECASE)
                if m2:
                    ans = m2.group(1).strip()
                    ans = re.sub(r'^\$+|\$+$', '', ans).strip()
                    return ans
                # otherwise use last non-empty line of tail
                lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
                if lines:
                    cand = lines[-1]
                    cand = re.sub(r'^[=:]\s*', '', cand).strip()
                    cand = re.sub(r'^\$+|\$+$', '', cand).strip()
                    return cand
        except Exception:
            pass

        # 3) last non-empty line of whole response
        try:
            lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
            if lines:
                last = lines[-1]
                # If it looks numeric or latex-like, return it
                if re.search(r'[\d\\\%\+\-\^\./eE]', last):
                    cand = re.sub(r'^[=:]\s*', '', last).strip()
                    cand = re.sub(r'^\$+|\$+$', '', cand).strip()
                    return cand
        except Exception:
            pass

        # 4) try to find last math-like token group (e.g., $-10^{4} \\mathrm{A}/\\mathrm{s}$ or scientific)
        try:
            math_pattern = r"(\$?[+-]?\d+(\.\d+)?([eE][+-]?\d+)?[A-Za-z0-9\\\^\{\}\_\-\+\*/]*\$?)"
            matches = re.findall(math_pattern, response)
            if matches:
                last_match = matches[-1][0]
                last_match = re.sub(r'^\$+|\$+$', '', last_match).strip()
                return last_match
        except Exception:
            pass

        # 5) fallback: return whole response
        return response

    def _quick_compare(self, response_expr, answer_expr, tol=1e-6):
        if response_expr is None or answer_expr is None:
            return False
        
        try:
            if response_expr.is_Number and answer_expr.is_Number:
                return abs(float(response_expr - answer_expr)) < tol
            if isinstance(response_expr, Eq) and isinstance(answer_expr, Eq):
                return simplify(response_expr.lhs - response_expr.rhs) == simplify(answer_expr.lhs - answer_expr.rhs)
            return simplify(response_expr - answer_expr) == 0
        except Exception:
            return False

    def _post_check(self, line: dict, prefetch: bool = False) -> Any:
        ans = line.get('answer', '')
        try:
            res = self._extract_answer_by_rule(line)
        except Exception:
            return False

        if str(res).strip() == str(ans).strip():
            return str(res).strip() if prefetch else True


        try:
            parsed_res = parse_latex(res)
            parsed_ans = parse_latex(ans)
            if self._quick_compare(parsed_res, parsed_ans):
                return latex(parsed_res) if prefetch else True
        except Exception:
            pass
        return False

    def Seephys_auxeval(self, line: dict) -> Dict[str, Any]:
        """
        Use the LLM judger to compare extracted answer with GT.
        Prefetch (fast path) tries symbolic/string match before calling judge LLM.
        """
        log = ""
        gt_answer = str(line.get("answer", ""))

        try:
            precheck = self._post_check(line, prefetch=True)
            if precheck is not False:
                return dict(log="Prefetch succeed (Symbolic Match)", res=1, extracted=precheck)
        except Exception as e:
            eval_logger.debug(f"Prefetch check error: {e}")

        extracted_answer = self._extract_answer_by_rule(line)
        if extracted_answer == "":
            
            log += "No extracted answer from model output.\n"
            return dict(log=log, res=0, extracted=extracted_answer)

        prompt = self.build_seephys_scoring_prompt(line, extracted_answer)
        for i in range(3):
            res_str = self.judger_generate(prompt, temperature=1, max_completion_tokens=4096, patience=2, sleep_time=0)
            if res_str == FAIL_MSG:
                log += f"Try {i}: judger call failed.\n"
                continue

            cleaned = str(res_str).strip()
            
            score_match = re.search(r'\b(0|1)\b', cleaned)
            if score_match:
                score = int(score_match.group(1))
                return dict(log=f"LLM judger returned: {cleaned}", res=score, extracted=extracted_answer)
            else:
                log += f"Try {i}: could not parse score from judger output: {cleaned}\n"
                continue

        log += "All retries failed; returning 0.\n"
        return dict(log=log, res=0, extracted=extracted_answer)

    def Seephys_process_line(self, line: dict) -> Dict[str, Any]:
        """
        Quick rule-based checking for cases where we can avoid LLM judger.
        Returns dict with match (1/0) and extracted value.
        """
        try:
            match = self._post_check(line, prefetch=False)
        except Exception as e:
            eval_logger.debug(f"Seephys_process_line _post_check error: {e}")
            match = False

        extracted = self._extract_answer_by_rule(line)
        return {
            "index": line.get("index", -1),
            "match": 1 if match else 0,
            "extracted": extracted,
            "gt": line.get("answer", ""),
        }
