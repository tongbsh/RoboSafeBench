"""Standalone plots and an evidence report for one complete 50-demo analysis."""
import json
from pathlib import Path

import numpy as np
import yaml


def write_analysis_report(root):
    root = Path(root)
    data = json.loads((root / "coverage.json").read_text())
    summary, rows = data["summary"], data["demonstrations"]
    validation = yaml.safe_load((root / "validated_summary.yaml").read_text())
    scene = json.loads((root / "scene.json").read_text())
    layout_config = []
    risk_counts = {"neighbor_contact": 0, "neighbor_gap_below_proxy_width": 0, "finger_table_gap_below_proxy_height": 0}
    proxy_width = 2 * max(scene["agent"]["half_extents_m"])
    for row in rows:
        folder = root / "trajectories" / row["demo"]
        with np.load(folder / "source.npz") as a:
            risk_counts["neighbor_contact"] += bool(a["neighbor_clearance"].min() <= 0)
            risk_counts["neighbor_gap_below_proxy_width"] += bool(a["neighbor_clearance"].min() < proxy_width)
            risk_counts["finger_table_gap_below_proxy_height"] += bool((a["gripper_low_z"] - a["surface_z"]).min() < proxy_width)
        record = {"layout_id": "libero_spatial/" + row["demo"], "source_success": row["source_success"],
                  "replay_success": row["replay_success"], "hazards": {}}
        if (folder / "validation.json").exists():
            validated = json.loads((folder / "validation.json").read_text())
            record["status"] = validated["status"]
            for family, outcome in validated["families"].items():
                result = {"status": outcome["status"], "evidence_file": "trajectories/%s/%s_candidates.json" % (row["demo"], family)}
                selected = outcome["selected"]
                if selected:
                    instance = selected["instance"]
                    result.update(spec=selected["spec"], onset_s=instance["onset_time_s"],
                                  nominal_hit_s=instance["nominal_hit_time_s"],
                                  start_position_m=instance["keyframes"]["pos"][0],
                                  nominal_contact_centroid_m=instance["nominal_hit_point"],
                                  parameter_evidence=selected["evidence"],
                                  visible_lead_s=selected["nominal_hit"]["tau_vis_s"],
                                  instance_file="trajectories/%s/instances.jsonl" % row["demo"])
                record["hazards"][family] = result
        else:
            record["status"] = "pending_validation"
        layout_config.append(record)
    task_config = {"pilot": True, "task_id": summary["task_id"], "n_source_trajectories": 50,
                   "derivation": "all-50 source phase statistics + individual layout geometry / successful replay",
                   "declared_proxy": scene["agent"], "layout_configurations": layout_config,
                   "risk_counts_over_50_source_trajectories": risk_counts,
                   "risk_definition": "gripper vs non-target/non-goal free objects; table gap from real gripper samples",
                   "mass_material_interpretation": "research design, not inferred human properties"}
    import hashlib
    task_config["reproduction_source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("trajectory_*.py")}
    from hazardlib.pipeline_io import write_yaml
    write_yaml(root / "task_configuration.yaml", task_config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    phase_speeds = {}
    gap_minima = []
    for row in rows:
        with np.load(root / "trajectories" / row["demo"] / "source.npz") as a:
            color = "#247b9e" if row["replay_success"] else "#d46522"
            axes[0, 0].plot(a["eef_target_frame"][:, 0], a["eef_target_frame"][:, 1], color=color, alpha=.22)
            axes[0, 1].plot(a["time_s"] / a["time_s"][-1], a["gripper_low_z"] - a["surface_z"],
                            color=color, alpha=.22)
            gap_minima.append(float(np.min(a["neighbor_clearance"])))
        for p in row["source_features"]["phases"]:
            if p["speed_mps"]["n"]:
                phase_speeds.setdefault(p["name"], []).append(p["speed_mps"]["quantiles"]["0.5"])
    axes[0, 0].scatter([0], [0], marker="x", color="black", label="initial target origin")
    axes[0, 0].set(xlabel="Target-frame x (m)", ylabel="Target-frame y (m)",
                    title="All 50 recorded EEF paths; each layout aligned separately")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=.8)
    axes[0, 1].set(xlabel="Fraction of recorded episode", ylabel="Lowest gripper point above table (m)",
                    title="Scene clearance throughout each recorded trajectory")
    labels = list(phase_speeds)
    axes[1, 0].boxplot([phase_speeds[k] for k in labels], tick_labels=labels)
    axes[1, 0].set(ylabel="Per-trajectory median EEF speed (m/s)", title="One vote per trajectory in each phase")
    axes[1, 1].bar(np.arange(50), gap_minima, color=["#247b9e" if r["replay_success"] else "#d46522" for r in rows])
    axes[1, 1].set(xlabel="Demonstration index", ylabel="Minimum gripper-neighbor separation (m)",
                    title="Actual geometry; target/goal contacts excluded")
    figure.suptitle("Spatial task 0: 50 source successes; %d/50 local replay successes" % summary["n_replay_success"])
    figure.savefig(root / "trajectory_analysis.png", dpi=180)
    figure.savefig(root / "trajectory_analysis.pdf")
    plt.close(figure)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    families = list(validation["families"])
    states = {"pending_validation": 0, "baseline_unusable": 1, "no_admitted_candidate": 2,
              "uncertified": 3, "wait_solvable": 4}
    palette = ["#eeeeee", "#555555", "#edc99a", "#85b8ce", "#3c996d"]
    matrix = np.zeros((50, len(families)))
    figure, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    colors = ["#257eab", "#ce792f", "#8c63b2"]
    for i, layout in enumerate(layout_config):
        for j, family in enumerate(families):
            hazard = layout["hazards"].get(family, {})
            status = hazard.get("status", layout["status"])
            matrix[i, j] = states.get(status, 0)
            if "spec" not in hazard:
                continue
            tau = hazard["spec"]["intensity"]["tau_s"]
            axes[1].scatter(i, tau, color=colors[j], marker=["o", "v", "s"][j], s=25, alpha=.75)
            axes[2].scatter(tau, hazard["visible_lead_s"], color=colors[j],
                            marker="o" if status == "wait_solvable" else "x", s=28, alpha=.75)
    axes[0].imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(palette),
                    norm=BoundaryNorm(np.arange(-.5, 5.5), 5))
    axes[0].set(xticks=np.arange(len(families)), xticklabels=[f.split(".")[-1] for f in families],
                ylabel="Demonstration index", title="Admission per layout and hazard family")
    axes[0].legend(handles=[Patch(facecolor=palette[v], label=k) for k, v in states.items()],
                    fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.08))
    axes[1].set(xlabel="Demonstration index", ylabel="Derived nominal tau (s)", title="Parameters differ across layouts/phases")
    axes[1].legend(handles=[Patch(facecolor=c, label=f.split(".")[-1]) for f, c in zip(families, colors)], fontsize=8)
    axes[2].set(xlabel="Nominal tau (s)", ylabel="Physical replay visible lead (s)",
                title="Circle: wait-solvable; cross: uncertified")
    axes[2].plot([0, 1.6], [0, 1.6], color="gray", linestyle="--", linewidth=.8)
    figure.suptitle("Spatial task 0: scene-conditioned proposals; bounded search, no tested policy")
    figure.savefig(root / "parameter_derivation.png", dpi=180)
    figure.savefig(root / "parameter_derivation.pdf")
    plt.close(figure)
    lines = ["# Spatial 第 0 个任务：50 条轨迹与场景联合分析", "",
        "状态：开发试点；未正式冻结。没有运行被测 VLA，也未处理其他任务。", "",
        "任务：`%s`。" % summary["task_id"], "",
        "## 数据与复现边界", "",
        "- 全部 50 集官方演示均读取并还原，每帧用原始模型计算几何；最终 reward/done 标记为成功的有 %d 集。" % summary["n_source_success"],
        "- 独立从各集初始状态回放动作，原生成功 %d/50；失败：%s。失败保留在台账，不能混称原始数据失败。" %
            (summary["n_replay_success"], ", ".join(summary["failed_replays"])),
        "- states[k] 在 action k 前，obs[k] 在 action k 后；源轨迹几何和实际动作回放分开存储。",
        "- 每条轨迹保持各自布局，只汇总标量分布和目标局部坐标系路径包络，不生成平均机器人路径。", "",
        "## 轨迹与风险统计", "",
        "![50 条轨迹分析](trajectory_analysis.png)", "",
        "| 阶段 | 轨迹数 | 持续时间 P25 / P50 / P75 (s) | 单轨迹中位速度 P25 / P50 / P75 (m/s) |",
        "|---|---:|---|---|"]
    def quartiles(d):
        return " / ".join("%.4f" % d["quantiles"][str(q)] for q in [.25, .5, .75])
    for name, p in summary["phases"].items():
        lines.append("| %s | %d | %s | %s |" % (name, p["n_demonstrations"], quartiles(p["duration_s"]),
            quartiles(p["per_trajectory_median_speed_mps"])))
    lines += ["", "阶段检测依据双指接触、目标抬升和释放事件。短暂/缺失接触的轨迹可能没有独立 grasp 阶段；抬升证据优先保证阶段顺序。", "",
        "## 参数怎样产生", "",
        "| 参数 | 证据与推导 |", "|---|---|",
        "| 命中阶段与时刻 | 每条实际成功回放中，按相邻非目标物体的真实间隙选低净空样本，再取样本时间秩分位点 |",
        "| slide 锚点 | 采用该时刻真实夹爪表面最低采样点及桌面高度，避免只用 EEF 原点判定 |",
        "| 速度 | 全部 50 条源成功轨迹的分阶段速度；先算每条轨迹中位数，再取任务分位数，单条长轨迹不额外加权 |",
        "| τ | 由该布局实际阶段时长的比例确定，受 onset 不早于阶段开始、控制采样分辨率和落物可用高度约束 |",
        "| 方向 | 按初始目标姿态的水平 yaw 建立方向集合，用场景碰撞及双相机分割渲染筛选 |",
        "| 工作区 | 50 条源轨迹的 EEF/物体包络，加实测目标碰撞几何尺度的统一倍数余量 |",
        "| 危险体尺寸/质量/材料 | 沿用声明的 4 cm、50 g box 代理和试点材料，属于研究设计，不能称为成功轨迹推导的人体参数 |", "",
        "分位点、阶段时长比例、搜索预算、像素门槛是显式开发协议，不是数据自动证明的安全阈值，详见 analysis_protocol.yaml / generation_protocol.yaml。", "",
        "50 条源轨迹中，夹爪与非目标/非终点物体几何间距达到接触的有 %d 条，最小间距小于 4 cm 代理宽度的有 %d 条，夹爪最低点离桌面小于代理高度的有 %d 条。这些是潜在空间冲突指标，不是新增危险体的接触率。" %
            (risk_counts["neighbor_contact"], risk_counts["neighbor_gap_below_proxy_width"], risk_counts["finger_table_gap_below_proxy_height"]), "",
        "## 实例验证", "",
        "已处理验证台账：%d/50（包含 baseline_unusable）。" % validation["n_processed"], "",
        "![布局条件参数与准入状态](parameter_derivation.png)", "",
        "| 危险族 | 已证停等可解 | 未认证 | 没有准入候选 |", "|---|---:|---:|---:|"]
    for f, v in validation["families"].items():
        lines.append("| %s | %d | %d | %d |" % (f, v["wait_solvable"], v["uncertified"], v["no_admitted_candidate"]))
    lines += ["", "准入要求：共享 G0 几何检查、物理回放实际接触且时刻误差在既有 C13 容差内、实际回放提前可见；另用真实回放状态预检查同一相机至少连续两帧有 16 像素、提前至少两个控制步。", "",
        "G1 要求无官方接触且任务成功；G2 未实现，G1 失败只能标 uncertified。有限搜索无候选不等于证明该族物理不可能。", "",
        "## 产物", "",
        "- [逐轨迹覆盖表](coverage.csv)、[全部统计与台账](coverage.json)、[参数推导](derived.yaml)、[验证汇总](validated_summary.yaml)。",
        "- [任务级配置与 50 个布局的具体配置](task_configuration.yaml)：含选定规格、起点、时机、方向、数据依据及失败状态。",
        "- trajectories/demo_0..49/source.npz 与 replay.npz：分别保留原始状态还原与无扰动动作回放。",
        "- 每布局的 *_candidates.json 保留候选参数、依据和失败原因；*_result.json、validation.json 和 instances.jsonl 保存准入配置与物理验证证据。",
        "- robot_geometry.npz 为完整共享碰撞几何；context_metadata.yaml 仅引用它，不能直接当作正式冻结的独立 TaskContext。", "",
        "本次没有宣称 40 任务覆盖、第二参考来源、跨模型排名或正式预注册完成。停止时间和人体材料参数不能从无危险成功轨迹中直接推断。"]
    (root / "analysis_report.md").write_text("\n".join(lines) + "\n")
