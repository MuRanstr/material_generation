from pathlib import Path

from project.generate_stable_materials import MaterialGenerator, Structure


def custom_evaluate_material(cif_path: str):
    print(f"评估材料: {cif_path}")

    # 1. 初始化生成器
    # 使用 dummy 路径，并设置一个临时输出目录
    output_dir = "temp_custom_results"
    Path(output_dir).mkdir(exist_ok=True)
    generator = MaterialGenerator(checkpoint_path="dummy", output_dir=output_dir)

    # 2. 分析材料（获取稳定性数据）
    try:
        original_df = generator.analyze_materials([cif_path])
    except Exception as e:
        print(f"错误: 分析材料失败 - {e}")
        return

    if original_df.empty:
        print("错误: 无法从 CIF 文件中解析结构。")
        return

    # 3. 预测 HER 性能（获取 ΔG）
    try:
        structure = Structure.from_file(cif_path)
        her_result = generator.her_predictor.predict_her_comprehensive(structure)
        # 提取 HER ΔG (eV)
        her_delta_g = her_result['co2rr_result'].get('e_h_adsorption_ev', None)

    except Exception as e:
        print(f"错误: HER 性能预测失败 - {e}")
        her_delta_g = 'Error'
        her_result = {'comprehensive_her_score': 0.0}  # 确保有默认值

    # 4. 计算 Stability Score (Normalized Hull Energy)
    hull_energy = original_df.iloc[0]['hull_energy']
    # 使用脚本中的归一化公式: 1 - E_hull / 0.5
    stability_score = 1 - hull_energy / 0.5

    # 5. 判断是否通过筛选
    passed_screening = (
            (original_df.iloc[0]['hull_energy'] < generator.stability_criteria['hull_energy_threshold']) and
            (original_df.iloc[0]['formation_energy'] < generator.stability_criteria['formation_energy_threshold']) and
            (original_df.iloc[0]['num_elements'] <= generator.stability_criteria['max_elements']) and
            (original_df.iloc[0]['2d_score'] >= generator.stability_criteria['min_2d_score'])
    )

    # 6. 打印结果
    print("\n--- 提取的评估指标 ---")
    print(f"材料公式: {original_df.iloc[0]['formula']}")
    print(f"1. Avg HER ΔG (eV): {her_delta_g if her_delta_g is not None else 'N/A'} eV")
    print(f"2. Stability Score: {stability_score:.3f} (Normalized Hull Energy)")
    print(f"3. 是否通过筛选: {'是' if passed_screening else '否'}")
    print(f"   (原始 Hull Energy: {hull_energy:.3f} eV/atom)")
    print(f"   (原始 Formation Energy: {original_df.iloc[0]['formation_energy']:.3f} eV/atom)")
    print(f"   (原始 2D Score: {original_df.iloc[0]['2d_score']:.3f})")

    # 清理临时文件
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
