from __future__ import annotations

from statistics import mean, median

import numpy as np


FEATURE_LABELS = {
    "en": {
        "area_fraction": "area fraction",
        "circularity": "circularity",
        "bbox_fill_ratio": "bounding-box fill",
        "aspect_ratio": "aspect ratio",
        "mean_brightness": "mean brightness",
        "contrast": "internal contrast",
        "darkness_p90": "darkness P90",
        "edge_density": "edge density",
    },
    "zh": {
        "area_fraction": "图像面积占比",
        "circularity": "圆度",
        "bbox_fill_ratio": "外接框填充率",
        "aspect_ratio": "长宽比",
        "mean_brightness": "平均亮度",
        "contrast": "内部对比度",
        "darkness_p90": "暗度 P90",
        "edge_density": "内部边缘密度",
    },
}


class EvidenceReportAgent:
    """Generate an auditable narrative from model outputs and measured evidence only."""

    def generate(self, samples: list[dict], failures: list[dict], language: str = "en") -> str:
        language = "zh" if language == "zh" else "en"
        return self._generate_zh(samples, failures) if language == "zh" else self._generate_en(samples, failures)

    @staticmethod
    def _groups(samples: list[dict]):
        ordered = sorted(samples, key=lambda item: item["viability"], reverse=True)
        group_count = max(1, min(5, len(ordered) // 3 or 1))
        return ordered, group_count, ordered[:group_count], ordered[-group_count:]

    @staticmethod
    def _correlations(samples: list[dict], language: str) -> list[str]:
        labels = FEATURE_LABELS[language]
        correlations = []
        if len(samples) >= 4 and np.std([item["viability"] for item in samples]) > 0:
            viability = np.asarray([item["viability"] for item in samples], dtype=float)
            for key, label in labels.items():
                feature = np.asarray([item["evidence"][key] for item in samples], dtype=float)
                if np.std(feature) <= 1e-8:
                    continue
                corr = float(np.corrcoef(viability, feature)[0, 1])
                if np.isfinite(corr) and abs(corr) >= 0.35:
                    if language == "en":
                        sentence = f"- Within-batch Pearson correlation between {label} and predicted viability: r = {corr:.2f}."
                    else:
                        sentence = f"- {label}与预测活性的批次内 Pearson r = {corr:.2f}。"
                    correlations.append((abs(corr), sentence))
        return [sentence for _, sentence in sorted(correlations, reverse=True)[:4]]

    def _generate_en(self, samples: list[dict], failures: list[dict]) -> str:
        if not samples:
            return "# Batch viability analysis report\n\nNo samples were analyzed successfully in this batch."
        ordered, group_count, top, bottom = self._groups(samples)
        values = [item["viability_percent"] for item in ordered]
        high = sum(item["viability"] >= 0.8 for item in ordered)
        medium = sum(0.6 <= item["viability"] < 0.8 for item in ordered)
        low = sum(item["viability"] < 0.6 for item in ordered)
        lines = [
            "# Batch organoid viability analysis report", "", "## Batch overview", "",
            f"- Successfully analyzed: {len(samples)}; failed: {len(failures)}.",
            f"- Viability range: {min(values):.1f}%–{max(values):.1f}%; mean: {mean(values):.1f}%; median: {median(values):.1f}%.",
            f"- High viability: {high}; intermediate viability: {medium}; low viability: {low}.",
            "", "## Representative samples", "",
            f"- Highest prediction: {ordered[0]['filename']} ({ordered[0]['viability_percent']:.1f}%).",
            f"- Lowest prediction: {ordered[-1]['filename']} ({ordered[-1]['viability_percent']:.1f}%).",
            "", f"## Image-evidence comparison (top and bottom {group_count})", "",
        ]
        observations = []
        for key, label in FEATURE_LABELS["en"].items():
            top_value = mean(item["evidence"][key] for item in top)
            bottom_value = mean(item["evidence"][key] for item in bottom)
            delta = top_value - bottom_value
            if abs(delta / max(abs(bottom_value), 1e-6)) >= 0.08:
                direction = "higher" if delta > 0 else "lower"
                observations.append(f"- Mean {label} was {direction} in the high-viability group ({top_value:.3f} vs {bottom_value:.3f}).")
        lines.extend(observations or ["- No marked differences were observed in the current morphology features between the high- and low-viability groups."])
        lines += ["", "## Within-batch correlation signals", ""]
        lines.extend(self._correlations(samples, "en") or ["- The sample size was insufficient, or no linear correlation with |r| ≥ 0.35 was identified."])
        qc = [item for item in samples if item["segmentation"]["yolo_confidence"] < 0.6 or item["segmentation"]["sam_score"] < 0.85]
        lines += ["", "## Quality control", ""]
        lines.append("- Manual review recommended: " + ", ".join(item["filename"] for item in qc) + "." if qc else "- All successfully analyzed samples passed the current YOLO/SAM threshold checks.")
        if failures:
            lines.append("- Failed samples: " + ", ".join(item["filename"] for item in failures) + ".")
        lines += ["", "## Interpretation boundary", "", "These results describe associations between model predictions and measurable image features within this batch. They do not establish a biological cause of high or low viability. Validation using fluorescence assays, culture conditions, and independent replicates is recommended."]
        return "\n".join(lines)

    def _generate_zh(self, samples: list[dict], failures: list[dict]) -> str:
        if not samples:
            return "# 批量活性分析报告\n\n本批次没有成功完成分析的样本。"
        ordered, group_count, top, bottom = self._groups(samples)
        values = [item["viability_percent"] for item in ordered]
        high = sum(item["viability"] >= 0.8 for item in ordered)
        medium = sum(0.6 <= item["viability"] < 0.8 for item in ordered)
        low = sum(item["viability"] < 0.6 for item in ordered)
        lines = [
            "# 批量类器官活性分析报告", "", "## 批次概览", "",
            f"- 成功分析：{len(samples)} 张；失败：{len(failures)} 张。",
            f"- 活性范围：{min(values):.1f}%–{max(values):.1f}%；均值：{mean(values):.1f}%；中位数：{median(values):.1f}%。",
            f"- 高活性：{high} 张；中等活性：{medium} 张；低活性：{low} 张。",
            "", "## 代表样本", "",
            f"- 最高预测：{ordered[0]['filename']}（{ordered[0]['viability_percent']:.1f}%）。",
            f"- 最低预测：{ordered[-1]['filename']}（{ordered[-1]['viability_percent']:.1f}%）。",
            "", f"## 高低活性组图像证据比较（Top/Bottom 各 {group_count} 张）", "",
        ]
        observations = []
        for key, label in FEATURE_LABELS["zh"].items():
            top_value = mean(item["evidence"][key] for item in top)
            bottom_value = mean(item["evidence"][key] for item in bottom)
            delta = top_value - bottom_value
            if abs(delta / max(abs(bottom_value), 1e-6)) >= 0.08:
                observations.append(f"- 高活性组的{label}平均值{'更高' if delta > 0 else '更低'}（{top_value:.3f} vs {bottom_value:.3f}）。")
        lines.extend(observations or ["- 本批次高低组在当前形态特征上的差异不明显。"])
        lines += ["", "## 批次内相关性线索", ""]
        lines.extend(self._correlations(samples, "zh") or ["- 样本量不足或未发现绝对值 ≥ 0.35 的线性相关线索。"])
        qc = [item for item in samples if item["segmentation"]["yolo_confidence"] < 0.6 or item["segmentation"]["sam_score"] < 0.85]
        lines += ["", "## 质量控制", ""]
        lines.append("- 建议人工复核：" + "、".join(item["filename"] for item in qc) + "。" if qc else "- 所有成功样本均通过当前 YOLO/SAM 阈值检查。")
        if failures:
            lines.append("- 失败样本：" + "、".join(item["filename"] for item in failures) + "。")
        lines += ["", "## 解释边界", "", "以上内容描述模型预测及同批次图像特征之间的关联，不构成活性高低的生物学因果证明。建议结合荧光实验、培养条件和独立重复进行验证。"]
        return "\n".join(lines)
