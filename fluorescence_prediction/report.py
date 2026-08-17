from __future__ import annotations

from statistics import mean, median

import numpy as np


FEATURE_LABELS = {
    "area_fraction": "图像面积占比",
    "circularity": "圆度",
    "bbox_fill_ratio": "外接框填充率",
    "aspect_ratio": "长宽比",
    "mean_brightness": "平均亮度",
    "contrast": "内部对比度",
    "darkness_p90": "暗度 P90",
    "edge_density": "内部边缘密度",
}


class EvidenceReportAgent:
    """Generate an auditable narrative from model outputs and measured evidence only."""

    def generate(self, samples: list[dict], failures: list[dict]) -> str:
        if not samples:
            return "# 批量活性分析报告\n\n本批次没有成功完成分析的样本。"

        ordered = sorted(samples, key=lambda item: item["viability"], reverse=True)
        values = [item["viability_percent"] for item in ordered]
        high = [item for item in ordered if item["viability"] >= 0.8]
        medium = [item for item in ordered if 0.6 <= item["viability"] < 0.8]
        low = [item for item in ordered if item["viability"] < 0.6]
        group_count = max(1, min(5, len(ordered) // 3 or 1))
        top = ordered[:group_count]
        bottom = ordered[-group_count:]

        lines = [
            "# 批量类器官活性分析报告",
            "",
            "## 批次概览",
            "",
            f"- 成功分析：{len(samples)} 张；失败：{len(failures)} 张。",
            f"- 活性范围：{min(values):.1f}%–{max(values):.1f}%；均值：{mean(values):.1f}%；中位数：{median(values):.1f}%。",
            f"- 高活性：{len(high)} 张；中等活性：{len(medium)} 张；低活性：{len(low)} 张。",
            "",
            "## 代表样本",
            "",
            f"- 最高预测：{ordered[0]['filename']}（{ordered[0]['viability_percent']:.1f}%）。",
            f"- 最低预测：{ordered[-1]['filename']}（{ordered[-1]['viability_percent']:.1f}%）。",
            "",
            f"## 高低活性组图像证据比较（Top/Bottom 各 {group_count} 张）",
            "",
        ]

        observations = []
        for key, label in FEATURE_LABELS.items():
            top_value = mean(item["evidence"][key] for item in top)
            bottom_value = mean(item["evidence"][key] for item in bottom)
            delta = top_value - bottom_value
            scale = max(abs(bottom_value), 1e-6)
            relative = delta / scale
            if abs(relative) >= 0.08:
                direction = "更高" if delta > 0 else "更低"
                observations.append(f"- 高活性组的{label}平均值{direction}（{top_value:.3f} vs {bottom_value:.3f}）。")
        lines.extend(observations or ["- 本批次高低组在当前形态特征上的差异不明显。"])

        lines.extend(["", "## 批次内相关性线索", ""])
        correlations = []
        if len(samples) >= 4 and np.std([item["viability"] for item in samples]) > 0:
            viability = np.asarray([item["viability"] for item in samples], dtype=float)
            for key, label in FEATURE_LABELS.items():
                feature = np.asarray([item["evidence"][key] for item in samples], dtype=float)
                if np.std(feature) <= 1e-8:
                    continue
                corr = float(np.corrcoef(viability, feature)[0, 1])
                if np.isfinite(corr) and abs(corr) >= 0.35:
                    correlations.append((abs(corr), f"- {label}与预测活性的批次内 Pearson r = {corr:.2f}。"))
        lines.extend([text for _, text in sorted(correlations, reverse=True)[:4]] or ["- 样本量不足或未发现绝对值 ≥0.35 的线性相关线索。"])

        qc = [item for item in samples if item["segmentation"]["yolo_confidence"] < 0.6 or item["segmentation"]["sam_score"] < 0.85]
        lines.extend(["", "## 质量控制", ""])
        if qc:
            lines.append("- 建议人工复核：" + "、".join(item["filename"] for item in qc) + "。")
        else:
            lines.append("- 所有成功样本均通过当前 YOLO/SAM 阈值检查。")
        if failures:
            lines.append("- 失败样本：" + "、".join(item["filename"] for item in failures) + "。")

        lines.extend([
            "",
            "## 解释边界",
            "",
            "以上内容描述模型预测及同批次图像特征之间的关联，不构成活性高低的生物学因果证明。建议结合荧光实验、培养条件和独立重复进行验证。",
        ])
        return "\n".join(lines)
