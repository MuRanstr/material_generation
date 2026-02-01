import os
from pathlib import Path
import sys
import shutil


sys.path.append(str(Path(__file__).parent))

try:
    from generate_stable_materials import MaterialGenerator, Structure

    print("成功导入项目模块。")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


def filter_generated_cifs(input_dirs, output_count=10, output_cif_dir="filtered_cifs"):
    print(f"开始筛选目录: {input_dirs}")

    # 创建保存 CIF 的输出目录
    output_path = Path(output_cif_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 初始化生成器 (使用 dummy 路径)
    temp_output = "temp_filter_results"
    Path(temp_output).mkdir(exist_ok=True)
    generator = MaterialGenerator(checkpoint_path="dummy", output_dir=temp_output)

    cif_files = []
    for input_dir in input_dirs:
        cif_files.extend(list(Path(input_dir).glob("*.cif")))
    print(f"找到 {len(cif_files)} 个 CIF 文件。")

    results = []

    for cif_path in cif_files:
        print(f"正在处理: {cif_path.name}...")
        try:
            # 加载结构
            structure = Structure.from_file(str(cif_path))

            # 3. 预测 HER 性能（获取 ΔG）
            her_result = generator.her_predictor.predict_her_comprehensive(structure)

            # 提取 HER ΔG (eV)
            her_delta_g = her_result['co2rr_result'].get('e_h_adsorption_ev', None)

            if her_delta_g is not None and her_delta_g != 'Error':
                # 获取稳定性数据
                original_df = generator.analyze_materials([str(cif_path)])

                if not original_df.empty:
                    row = original_df.iloc[0]

                    # 记录结果
                    results.append({
                        'file_name': cif_path.name,
                        'formula': row['formula'],
                        'deltaG': her_delta_g,
                        'hull_energy': row['hull_energy'],
                        'formation_energy': row['formation_energy'],
                        '2d_score': row['2d_score']
                    })
                    target_file = output_path / cif_path.name
                    shutil.copy(cif_path, target_file)

                    print(f"  [通过] deltaG: {her_delta_g} -> 已保存至 {output_cif_dir}")
                else:
                    print(f"  [跳过] 无法获取稳定性数据")
            else:
                print(f"  [跳过] deltaG 为 NaN 或 Error")

            if len(results) >= output_count:
                print(f"已达到目标筛选数量 ({output_count})，停止处理。")
                break

        except Exception as e:
            print(f"  [错误] 处理 {cif_path.name} 时出错: {e}")

    # 汇总显示
    if results:
        print(f"\n筛选出的 {len(results)} 个 CIF 文件已保存至目录: {output_cif_dir}")
    else:
        print("\n未找到符合条件的 CIF 文件。")

    # 清理临时目录
    shutil.rmtree(temp_output, ignore_errors=True)


if __name__ == "__main__":
    # 设定输入目录
    input_directories = [
        "generated_cifs",
    ]

    valid_dirs = [d for d in input_directories if os.path.exists(d)]
    if not valid_dirs:
        print(f"错误: 未找到任何有效的输入目录。")
    else:
        # 可以通过参数指定输出目录
        filter_generated_cifs(valid_dirs, output_count=10, output_cif_dir="last_ten_structure")