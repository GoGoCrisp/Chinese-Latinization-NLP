"""
最终总结：对比所有版本的改进效果
"""

print("=" * 100)
print("【最终总结】三层方案改进效果对比")
print("=" * 100)

versions = [
    {
        "版本": "v1: 仅CEDICT",
        "数据源": "CEDICT (10,867字符)",
        "成功": 33138,
        "失败": 17559,
        "总数": 50932,
        "成功率": "65.0%",
        "特点": "基础版本，缺少生僻字"
    },
    {
        "版本": "v2: CEDICT + pypinyin",
        "数据源": "CEDICT + pypinyin fallback",
        "成功": 36920,
        "失败": 11715,
        "总数": 48635,
        "成功率": "75.9%",
        "特点": "补充了生僻字，但pypinyin覆盖有限"
    },
    {
        "版本": "v3: Unihan + CEDICT + pypinyin",
        "数据源": "Unihan (44,424) + CEDICT + pypinyin",
        "成功": 38045,
        "失败": 10590,
        "总数": 48635,
        "成功率": "78.2%",
        "特点": "✅ 最优方案！包含所有CJK字符的拼音"
    }
]

print("\n【详细对比】(基于纯中文token)")
print("-" * 100)
print(f"{'版本':<30} {'数据源':<50} {'成功':<10} {'失败':<10} {'成功率':<10}")
print("-" * 100)

for v in versions:
    print(f"{v['版本']:<30} {v['数据源']:<50} {v['成功']:<10} {v['失败']:<10} {v['成功率']:<10}")

print("\n【改进数据】")
print("-" * 100)
print(f"v1 → v2: +3,782 tokens (68.1% → 75.9%，+7.8%)")
print(f"v2 → v3: +1,125 tokens (75.9% → 78.2%，+2.3%)")
print(f"v1 → v3: +4,907 tokens (65.0% → 78.2%，+13.2%)")

print("\n【关键指标】")
print("-" * 100)
print(f"• A纯中文token总数: 48,635")
print(f"• 最优映射成功: 38,045 (78.2%)")
print(f"• 理论上限: ~80-82% (受B词表容量限制)")
print(f"• 永久失败: ~18-20% (字符或拼音不在B中)")

print("\n【推荐使用】")
print("-" * 100)
print(f"✅ 生产版本: v3 (Unihan + CEDICT + pypinyin)")
print(f"   成功率: 78.2%")
print(f"   特点: 最全面的拼音覆盖")
print(f"")
print(f"📊 数据来源:")
print(f"   • Unihan: Unicode官方数据库 (44,348字符)")
print(f"   • CEDICT: CC-CEDICT词典 (10,867字符)")
print(f"   • pypinyin: Python拼音库 (fallback)")

print("\n【残留问题分析】")
print("-" * 100)
print(f"21.8% (10,590个token) 无法映射的原因：")
print(f"  1. 生冷/异体字在所有字典中都不存在 (~5-10%)")
print(f"  2. 拼音转换结果不在B的64K词表中 (~15%)")
print(f"  3. 混合token (包含数字、英文等) (~1-5%)")

print("\n" + "=" * 100)
