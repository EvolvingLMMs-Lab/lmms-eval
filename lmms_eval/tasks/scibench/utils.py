import re
from math import isclose
from typing import Dict, List, Tuple

FEWSHOT_PROMPT = """Problem:
Suppose that $10.0 \mathrm{~mol} \mathrm{C}_2 \mathrm{H}_6(\mathrm{~g})$ is confined to $4.860 \mathrm{dm}^3$ at $27^{\circ} \mathrm{C}$. Predict the pressure exerted by the ethane from the perfect gas.

Solution:
To predict ethane's pressure using the ideal gas law (PV = nRT), I'll convert temperature from 27°C to Kelvin (300.15 K, rounded to 300 K for significant figures), then substitute the given values: 10.0 mol of ethane, volume of 4.860 L, gas constant R = 0.0821 L·atm/(mol·K), and T = 300 K. Rearranging the ideal gas equation to P = nRT/V and calculating: P = (10.0 mol × 0.0821 L·atm/(mol·K) × 300 K) ÷ 4.860 L = 246.3 L·atm ÷ 4.860 L ≈ 50.7 atm. Since temperature has two significant figures, the final pressure is \(\boxed{50.7}\) atm.
Final Answer: The final answer is \(\boxed{50.7}\). I hope it is correct.

Problem:
Assume that all gases are perfect and that data refer to 298.15 K unless otherwise stated. Calculate the change in chemical potential of a perfect gas when its pressure is increased isothermally from $1.8 \mathrm{~atm}$ to $29.5 \mathrm{~atm}$ at $40^{\circ} \mathrm{C}$.

Solution:
To determine the change in chemical potential (Δμ) of a perfect gas during pressure change, I begin with the fundamental relation μ = μ° + RT ln(P/P°), which yields Δμ = RT ln(Pf/Pi) for changes between two states. Converting the given temperature of 40°C to 313.15K and using R = 8.314 J/(mol·K), I calculate Δμ = (8.314 J/(mol·K))(313.15K)ln(29.5/1.8). The pressure ratio 29.5/1.8 ≈ 16.39 gives ln(16.39) ≈ 2.797, so Δμ = 8.314 × 313.15 × 2.797 ≈ 7274.5 J/mol, which rounds to 7.3 kJ/mol.
Final Answer: The final answer is \(\boxed{7.3}\). I hope it is correct.

Problem:
Show that the small angular deviation of $\epsilon$ of a plumb line from the true vertical (i.e., toward the center of Earth) at a point on Earth's surface at a latitude $\lambda$ is $\epsilon = \frac{R\omega^2sin\lambda cos\lambda}{g_0 - R\omega^2 cos^2\lambda}$ where R is the radius of Earth. What is the value (in seconds of arc) of the maximum deviation? Note that the entire denominator in the answer is actually the effective $g$, and $g_0$ denotes the pure gravitational component.

Solution:
To determine the small angular deviation (ε) of a plumb line from true vertical due to Earth's rotation, we analyze the balance of forces at latitude λ: gravitational force (Fg = mg0) toward Earth's center and centrifugal force (Fc = mRω²cosλ) perpendicular to the rotation axis. The centrifugal force resolves into a vertical component (Fc,v = mRω²cos²λ) that reduces effective gravity to g = g0 - Rω²cos²λ, and a horizontal component (Fc,h = mRω²sinλcosλ) pulling toward the equator. The angular deviation equals the ratio of horizontal force to effective gravity: ε = Rω²sinλcosλ/(g0 - Rω²cos²λ). To find maximum deviation, we differentiate with respect to λ and find it occurs at λ = 45°. Using Earth values (R = 6.371×10⁶ m, ω = 7.292×10⁻⁵ rad/s, g0 ≈ 9.81 m/s²), we calculate the numerator at 45° as 1.697×10⁻² m/s² and denominator as 9.793 m/s², yielding εmax = 1.733×10⁻³ rad or approximately 357 arcseconds (6 arcminutes).
Final Answer: The final answer is \(\boxed{6}\). I hope it is correct."""


def scibench_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = doc["problem_text"]
    if doc["unit"].strip():
        question = question + " The unit of the answer is " + doc["unit"] + "."
    return f"{pre_prompt}{question}{post_prompt}"


def extract_boxed_answers(text):
    # Find all boxed contents
    matches = re.findall(r"boxed{([^}]*)}", text)
    for m in matches:
        # Strip spaces
        candidate = m.strip()
        # Keep only the numeric ones (int or decimal, with optional sign)
        if re.fullmatch(r"[-+]?\d*\.?\d+", candidate):
            return candidate
    return None


def remove_not(x):
    match_number = re.compile("[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?")
    result = re.findall(match_number, x)
    if len(result) != 0:
        return re.split(match_number, x)[-1]
    return None


def cal_not(inputs):
    try:
        x, ab = list(inputs)
        match_number = re.compile("10\^[{]?\ *-?[0-9]+\ *[}]?")
        ab = re.findall(match_number, ab)[0]
        ab = ab[ab.find("^") + 1 :]
        if "{" in ab:
            ab = ab[ab.find("{") + 1 :]
        if "}" in ab:
            ab = ab[: ab.find("}")]
        x = x.strip()
        out = float(x) * 10 ** float(ab)
        # print(float(x)*10**float(ab))
        return str(out)
    except:
        print("error")
    return inputs


def parse_not(inputs):
    try:
        if not inputs:
            return "", ""
        if "\\times" in inputs:
            x, ab = inputs.split("\\times")
        elif "\times" in inputs:
            x, ab = inputs.split("\times")
        elif "*" in inputs:
            x, ab = inputs.split("*")
        else:
            return inputs
        return x, ab
    except:
        return "", ""


def equiv_with_unit(model_output, answer, unit):
    model_output = model_output.replace(",", "")
    print("Model_output: ", model_output)
    try:
        ans = float(answer.strip())
        first = isclose(float(model_output.strip()), ans, rel_tol=0.05)
    except:
        first = False
    try:
        model = model_output.strip().split()[0]
        second = isclose(float(model.strip()), ans, rel_tol=0.05)
    except:
        second = False
    if first or second:
        return True
    return False


def clean_number_string(s):
    return s.replace(",", "").replace("−", "-").strip()


def scibench_process_results(doc: Dict, result: List[str]) -> Dict[str, float]:
    pred = result[0]
    pred = extract_boxed_answers(pred)
    if pred:
        res_equiv = isclose(float(clean_number_string(pred)), float(clean_number_string(doc["answer_number"])), rel_tol=0.05)
        score = 1 if res_equiv else 0
    else:
        score = 0
    return {"accuracy": score}


def scibench_multishot_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = doc["problem_text"]
    if doc["unit"].strip():
        question = question + " The unit of the answer is " + doc["unit"] + "."
    return FEWSHOT_PROMPT + "\n" + question + "\nAnswer: Let's think step by step."
