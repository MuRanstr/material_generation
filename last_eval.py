import os
import glob
import pandas as pd
import io
import numpy as np
from contextlib import redirect_stdout
from project.generate_stable_materials import MaterialGenerator
from project.utils.geo_utils import custom_evaluate_material


def run_comprehensive_evaluation(cif_directory):
    """
    对指定目录下的 CIF 文件进行全面评估，包括：
    1. 合成性评分 (调用 MaterialGenerator 逻辑)
    2. Delta G (调用 energy_process 逻辑)
    3. Stability Score (调用 energy_process 逻辑)
    """
    print(f"开始全面性能评估 (目录: {cif_directory})")

    # 1. 初始化 MaterialGenerator (用于合成性评分)
    generator = MaterialGenerator(
        checkpoint_path="dummy_path",
        output_dir="eval_results"
    )

    # 2. 获取所有待评估的 CIF 文件
    cif_files = glob.glob(os.path.join(cif_directory, "*.cif"))
    if not cif_files:
        print(f"错误: 在 {cif_directory} 目录下未找到 CIF 文件。")
        return

    print(f"找到 {len(cif_files)} 个待评估文件。")

    results = []

    for cif_path in cif_files:
        formula = os.path.basename(cif_path).split('_')[0]

        # A. 计算合成性评分 (Synthesis Score)
        try:
            analysis = generator._analyze_single_material(cif_path)
            synthesis_score = analysis.get('synthesis_score', 0.0)
            formula = analysis.get('formula', formula)
        except Exception as e:
            print(f"计算 {cif_path} 的合成性评分时出错: {e}")
            synthesis_score = 0.0

        # B. 计算 Delta G 和 Stability Score
        delta_g = None
        stability_score = 0.0

        f = io.StringIO()
        with redirect_stdout(f):
            try:
                custom_evaluate_material(cif_path)
                output = f.getvalue()

                for line in output.split('\n'):
                    if "Avg HER ΔG (eV):" in line:
                        val_str = line.split(':')[1].strip().split(' ')[0]
                        if val_str not in ['N/A', 'Error']:
                            delta_g = float(val_str)
                    if "Stability Score:" in line:
                        try:
                            ss_val = float(line.split(':')[1].split('(')[0].strip())
                            stability_score = ss_val
                        except:
                            pass
            except Exception as e:
                print(f"评估 {cif_path} 的性能时出错: {e}")

        results.append({
            'formula': formula,
            'synthesis_score': synthesis_score,
            'delta_g': delta_g,
            'stability_score': stability_score
        })

    # 3. 汇总结果并计算平均值
    df = pd.DataFrame(results)

    if not df.empty:
        print("\n--- 评估结果摘要 ---")
        print(df.to_string(index=False))

        # 计算平均值 (忽略 None/NaN)
        avg_synthesis = df['synthesis_score'].mean()
        avg_delta_g = df['delta_g'].dropna().mean()
        avg_stability = df['stability_score'].mean()

        print("\n" + "=" * 30)
        print(f"综合评估统计 (样本数: {len(df)}):")
        print(f"1. 平均合成性评分: {avg_synthesis:.4f}")
        print(f"2. 平均 Delta G (eV): {avg_delta_g:.4f}" if not np.isnan(avg_delta_g) else "2. 平均 Delta G: N/A")
        print(f"3. 平均稳定性得分: {avg_stability:.4f}")
        print("=" * 30)

    else:
        print("未获得任何有效的评估结果。")


if __name__ == "__main__":
    # 优先评估生成的 CIF 文件
    target_dir = "old_cif_files"

    run_comprehensive_evaluation(target_dir)