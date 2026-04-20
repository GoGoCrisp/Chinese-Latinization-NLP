"""
检查B中是否有lv相关的tokens
"""
import json

with open('tokenizers/pinyin_toneless_64k_train90.json', 'r', encoding='utf-8') as f:
    vocab_b = json.load(f)['model']['vocab']

print("=" * 100)
print("检查B中lv相关的tokens")
print("=" * 100)

# 查找所有包含'lv'的tokens
lv_tokens = [token for token in vocab_b.keys() if 'lv' in token]

print(f"\n【B中包含'lv'的token】")
print(f"总数: {len(lv_tokens):,}")
print(f"示例 (前50个):")
for token in sorted(lv_tokens)[:50]:
    print(f"  '{token}'")

# 具体查找
test_tokens = ['lv', 'lu', 'lü', 'nv', 'nü', 'lvan', 'lve', 'lvi', 'lvo', 'lvu']
print(f"\n【特定token查询】")
for token in test_tokens:
    print(f"  '{token}' 在B中: {token in vocab_b}")

