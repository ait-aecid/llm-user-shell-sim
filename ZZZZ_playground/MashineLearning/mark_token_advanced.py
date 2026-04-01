#!/usr/bin/env python3
"""
log_token_attribution_html.py

Train 3 models on log lines with TF-IDF word n-grams and generate an HTML report
that highlights, for each log line and each model, the top-K most influential
present n-grams toward the predicted class:

  top1 = red
  top2 = orange
  top3 = green

Works with:
  - LinearSVC
  - LogisticRegression
  - ComplementNB (naive bayes)

Notes:
- With ngram_range=(1,3) or (1,5), the highlighted "token" can be a multi-word
  span (e.g., "type=EXECVE user=root cmd=/usr/bin").
- Highlighting is done by searching for the n-gram substring in the original text.
  If an n-gram occurs multiple times, only the first occurrence is highlighted.

Install:
  pip install scikit-learn numpy
Run:
  python log_token_attribution_html.py
Output:
  attribution_report.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import html
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TopFeature:
    text: str
    contribution: float


@dataclass
class ModelAttribution:
    model: str
    pred: int
    decision: float
    top_features: List[TopFeature]
    highlighted_html: str


# -----------------------------
# Attribution helpers
# -----------------------------
def get_weights_for_attribution(clf) -> np.ndarray:
    """
    Return a weight vector w (n_features,) such that token contribution is
    approximated by w_j * x_j toward class 1.

    Supported:
      - LinearSVC / LogisticRegression (coef_)
      - ComplementNB / MultinomialNB (feature_log_prob_)
    """
    if hasattr(clf, "coef_"):
        return clf.coef_.ravel()

    if hasattr(clf, "feature_log_prob_"):
        logp = clf.feature_log_prob_
        if logp.shape[0] != 2:
            raise ValueError("Binary classification expected (2 classes).")
        return (logp[1] - logp[0]).ravel()

    raise TypeError(f"Unsupported model type for attribution: {type(clf).__name__}")


def decision_value(clf, X_row) -> float:
    """
    Scalar decision:
      - linear models: decision_function
      - NB: log-prob difference (class1 - class0)
    """
    if hasattr(clf, "decision_function"):
        return float(clf.decision_function(X_row)[0])

    if hasattr(clf, "predict_log_proba"):
        lp = clf.predict_log_proba(X_row)[0]
        return float(lp[1] - lp[0])

    # last resort (rare)
    proba = clf.predict_proba(X_row)[0]
    return float(proba[1] - proba[0])


def top_k_features_for_text(
    text: str,
    vectorizer: TfidfVectorizer,
    clf,
    k: int = 3,
) -> Tuple[int, float, List[TopFeature]]:
    """
    Compute top-K present features (n-grams) that most support the predicted class.

    We compute contributions toward class 1:
      contrib1_j = w_j * x_j

    If predicted class is 1: pick largest contrib1_j
    If predicted class is 0: pick largest (-contrib1_j) (i.e., most pushes to class 0)
    """
    X = vectorizer.transform([text])
    pred = int(clf.predict(X)[0])
    dec = decision_value(clf, X)

    w = get_weights_for_attribution(clf)
    feat_names = vectorizer.get_feature_names_out()

    nz = X.nonzero()[1]
    print("\n[VEC INFO]")
    print("text:", text)
    print("nonzero features:", nz.size)
    if nz.size == 0:
        return pred, dec, [TopFeature("<NO_FEATURES>", 0.0)]

    x_vals = X.data  # aligned with nz for a single CSR row
    contrib1 = w[nz] * x_vals

    if pred == 1:
        scores = contrib1
    else:
        scores = -contrib1

    # pick top-k indices by descending score
    order = np.argsort(-scores)[:k]

    top_feats: List[TopFeature] = []
    for idx in order:
        fidx = int(nz[idx])
        ftxt = str(feat_names[fidx])
        fscore = float(scores[idx])
        top_feats.append(TopFeature(ftxt, fscore))

    return pred, dec, top_feats


# -----------------------------
# HTML highlighting
# -----------------------------
def highlight_text_html(text: str, ranked_features: List[TopFeature]) -> str:
    """
    Highlight first occurrence of top1/top2/top3 in HTML.
    Uses true orange and includes a tooltip with contribution score.

    Important:
    - We escape the original text first.
    - Then we perform safe substring replacement on the escaped text by searching
      for the escaped feature text.
    - Only first occurrence per feature is highlighted.
    """
    styles = [
        "background:#ffe5e5;color:#b00000;font-weight:700;padding:0 2px;border-radius:3px;",  # red-ish
        "background:#fff1db;color:#b05a00;font-weight:700;padding:0 2px;border-radius:3px;",  # orange-ish
        "background:#e6ffe6;color:#006b00;font-weight:700;padding:0 2px;border-radius:3px;",  # green-ish
    ]

    escaped = html.escape(text)

    out = escaped
    for i, feat in enumerate(ranked_features[:3]):
        token = feat.text
        token_esc = html.escape(token)

        if not token_esc.strip():
            continue

        idx = out.find(token_esc)
        if idx == -1:
            # if the exact n-gram string isn't found (rare), skip it
            print("\n[HIGHLIGHT MISS]")
            print("  raw feature:", repr(token))
            print("  escaped feature:", repr(token_esc))
            print("  original text:", repr(text))
            print("  escaped text:", repr(escaped))
            continue

        tooltip = f"{token} | contribution={feat.contribution:+.6f}"
        tooltip_esc = html.escape(tooltip)

        span = (
            f"<span class='hl' style='{styles[i]}' title='{tooltip_esc}'>"
            f"{token_esc}</span>"
        )

        out = out[:idx] + span + out[idx + len(token_esc):]

    return out


def build_html_report(rows: List[Tuple[str, List[ModelAttribution]]], title: str) -> str:
    """
    rows: list of (original_log_line, per-model attributions)
    """
    css = """
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
           margin: 20px; color: #111; }
    h1 { margin: 0 0 8px 0; }
    .sub { color:#444; margin: 0 0 18px 0; }
    .logblock { border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; margin: 14px 0; }
    .logline { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
               background: #fafafa; padding: 10px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { text-align: left; padding: 8px; border-top: 1px solid #eee; vertical-align: top; }
    th { background: #fcfcfc; color: #333; font-weight: 600; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#f1f5f9; color:#0f172a; font-size: 12px; }
    .note { color:#555; font-size: 13px; margin-top: 12px; }
    """

    head = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="sub">
    Hover highlighted n-grams to see contribution. Top1=red, Top2=orange, Top3=green.
  </p>
"""

    blocks = []
    for i, (logline, attribs) in enumerate(rows, start=1):
        blocks.append(f"<div class='logblock'>")
        blocks.append(f"<div class='pill'>Log #{i}</div>")
        blocks.append(f"<div class='logline'>{html.escape(logline)}</div>")

        blocks.append("<table>")
        blocks.append("<tr><th>Model</th><th>Pred</th><th>Decision</th><th>Highlighted (top-3)</th><th>Top features</th></tr>")

        for a in attribs:
            top_list = "<br/>".join(
                f"<span class='mono'>{html.escape(f.text)}</span> "
                f"<span class='mono' style='color:#555'>(+{f.contribution:.6f})</span>"
                if f.contribution >= 0 else
                f"<span class='mono'>{html.escape(f.text)}</span> "
                f"<span class='mono' style='color:#555'>({f.contribution:.6f})</span>"
                for f in a.top_features[:3]
            )

            blocks.append(
                "<tr>"
                f"<td class='mono'>{html.escape(a.model)}</td>"
                f"<td class='mono'>{a.pred}</td>"
                f"<td class='mono'>{a.decision:+.6f}</td>"
                f"<td class='logline'>{a.highlighted_html}</td>"
                f"<td>{top_list}</td>"
                "</tr>"
            )

        blocks.append("</table>")
        blocks.append("</div>")

    tail = """
  <p class="note">
    Note: with word n-grams, highlighted spans can be multi-word. Highlighting finds the first exact match
    of the n-gram string in the original line.
  </p>
</body>
</html>
"""
    return head + "\n".join(blocks) + tail


def main():
    # -----------------------------
    # 1) Sample data (your logs)
    # -----------------------------
    texts = [
        # SYSCALL → label 1
        "type=SYSCALL success=yes exit=0 uid=33 comm=apache2",
        "type=SYSCALL success=yes exit=0 uid=0 comm=cron",
        "type=SYSCALL success=no exit=13 uid=33 comm=apache2",
        "type=SYSCALL success=no exit=1 uid=0 comm=sudo",
        "type=SYSCALL success=yes exit=0 uid=1000 comm=systemd",
        "type=SYSCALL success=no exit=5 uid=1000 comm=ssh",
        "type=SYSCALL success=yes exit=0 uid=33 comm=nginx",
        "type=SYSCALL success=no exit=13 uid=33 comm=php-fpm",
        "type=SYSCALL success=yes exit=0 uid=0 comm=rsyslogd",
        "type=SYSCALL success=no exit=1 uid=1000 comm=bash",
        "type=SYSCALL success=yes exit=0 uid=33 comm=mysql",
        "type=SYSCALL success=no exit=2 uid=0 comm=systemctl",
        "type=SYSCALL success=yes exit=0 uid=1000 comm=python",
        "type=SYSCALL success=no exit=13 uid=33 comm=node",

        # EXECVE → label 0
        "type=EXECVE user=root cmd=/usr/bin/apt update",
        "type=EXECVE user=root cmd=/usr/bin/systemctl restart apache2",
        "type=EXECVE user=www-data cmd=/usr/bin/php -v",
        "type=EXECVE user=root cmd=/bin/sh -c ls /root",
        "type=EXECVE user=alice cmd=/usr/bin/vim /etc/hosts",
        "type=EXECVE user=root cmd=/usr/bin/journalctl -xe",
        "type=EXECVE user=bob cmd=/usr/bin/python script.py",
        "type=EXECVE user=root cmd=/bin/sh -c rm -rf /tmp/test",
        "type=EXECVE user=www-data cmd=/usr/bin/curl http://localhost",
        "type=EXECVE user=root cmd=/usr/bin/docker ps",
        "type=EXECVE user=alice cmd=/usr/bin/ssh user@server",
        "type=EXECVE user=root cmd=/usr/bin/find /var/log",
        "type=EXECVE user=bob cmd=/usr/bin/ls -la",
        "type=EXECVE user=root cmd=/usr/bin/tail -f /var/log/syslog",
        "type=EXECVE user=unknown cmd=/bin/sh -c nc 10.0.0.5 4444",
    ]
    y = [1] * 14 + [0] * 15



    # -----------------------------
    # 2) Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=42, stratify=y
    )

    # -----------------------------
    # 3) Vectorizer
    # Choose (1,3) or (1,5)
    # -----------------------------
    NGRAM_MAX = 3  # <-- change to 3 if you want (1,3)

    '''
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(2, NGRAM_MAX),
        lowercase=False,
    )
    '''

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        token_pattern=None,     # important when using tokenizer
        ngram_range=(2, NGRAM_MAX),
        lowercase=False,
    )

    vectorizer.fit(texts)             # build vocabulary + IDF on all lines
    Xtr = vectorizer.transform(X_train)

    #Xtr = vectorizer.fit_transform(X_train)

    # -----------------------------
    # 4) Models
    # -----------------------------
    models: Dict[str, object] = {
        "LinearSVC": LinearSVC(C=1.0),
        "LogReg": LogisticRegression(max_iter=2000, C=1.0),
        "ComplementNB": ComplementNB(alpha=0.1),
    }

    for _, clf in models.items():
        clf.fit(Xtr, y_train)

    # -----------------------------
    # 5) Build HTML rows: per log line, per model
    # -----------------------------
    rows: List[Tuple[str, List[ModelAttribution]]] = []

    for line in X_test:
        per_model: List[ModelAttribution] = []
        for name, clf in models.items():
            pred, dec, top_feats = top_k_features_for_text(
                line, vectorizer, clf, k=3
            )
            highlighted = highlight_text_html(line, top_feats)
            per_model.append(
                ModelAttribution(
                    model=name,
                    pred=pred,
                    decision=dec,
                    top_features=top_feats,
                    highlighted_html=highlighted,
                )
            )
        rows.append((line, per_model))

    # -----------------------------
    # 6) Write HTML report
    # -----------------------------
    title = f"Log Attribution Report (TF-IDF word ngrams (1,{NGRAM_MAX}))"
    html_doc = build_html_report(rows, title)

    out_path = "playground/attribution_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"Wrote HTML report to: {out_path}")
    print("Open it in your browser. (Hover highlights for contribution scores.)")


if __name__ == "__main__":
    main()
