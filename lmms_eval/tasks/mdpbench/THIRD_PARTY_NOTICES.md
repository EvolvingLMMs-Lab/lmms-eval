# Third-party notices

The prediction parsing, element matching, and table normalization helpers are
adapted from the MIT-licensed MDPBench reference implementation. The license
text is included in `LICENSE-MDPBENCH`.

The Character Detection Matching (CDM) implementation under `cdm/` and
`cdm_metric.py` is adapted from OpenDataLab's UniMERNet CDM implementation,
licensed under Apache-2.0. The integration removes demo assets and applications,
adds bounded subprocess execution, and exposes the metric through lmms-eval.
The Apache-2.0 license text is included in `cdm/LICENSE`.

The CDM tokenizer contains code adapted from `harvardnlp/im2markup`, licensed
under MIT. Its license text is included in
`cdm/modules/tokenize_latex/LICENSE-IM2MARKUP`.

The tokenizer vendors a modified KaTeX 0.6.0 parser, licensed under MIT. Its
license text is included in
`cdm/modules/tokenize_latex/third_party/katex/LICENSE.txt`. The historical
unlicensed `match-at@0.1.0` dependency has been replaced with the equivalent
sticky regular-expression support built into modern Node.js.

TEDS is derived from IBM's PubTabNet implementation and retains its original
Apache-2.0 copyright and license header in `teds_metric.py`.
