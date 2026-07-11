# Korean KHDB Vocabulary N:1 Report

This is a tokenizer-vocabulary collision diagnostic. Mixed-script BPE
vocabulary tokens are converted to Hangulized surfaces with the same
Gukhanmun settings used in corpus preparation, then grouped by converted
surface form.

## Inputs

- mixed tokenizer: `4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_mixed_bpe_32k.json`
- Hangulized tokenizer: `4.Korean/korean_khdb_magazine_audit/data/tokenizers/korean_hangulized_bpe_32k.json`
- train mixed for frequency weighting: `4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.mixed_chunks_nospace.txt`
- dev mixed for frequency weighting: `4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt`

## Converter

- backend: gukhanmun
- version: gukhanmun 0.2.0
- command settings: `--rendering hangul-only --disambiguation off`
- cleanup: remove pure-Hanja parenthetical annotations, delete remaining Hanja, remove whitespace

## Debug Conversion

| input | output | leftover Hanja after cleanup |
|---|---|---:|
| 記事 | 기사 | 0 |
| 技師 | 기사 | 0 |
| 騎士 | 기사 | 0 |
| 會社 | 회사 | 0 |
| 社會 | 사회 | 0 |
| 國民 | 국민 | 0 |
| 權利 | 권리 | 0 |
| 權力 | 권력 | 0 |
| 大韓 | 대한 | 0 |
| 日本 | 일본 | 0 |

## Vocab Filtering

- mixed vocab size: `32000`
- Hangulized vocab size: `32000`
- mixed valid lexical tokens: `30897`
- mixed Hanja-containing tokens: `19999`
- Hangulized valid lexical tokens: `30708`
- Hangulized pure Hangul strict tokens: `25461`
- Hangulized pure Hangul loose tokens: `30708`

## Exact Overlap

- all valid exact overlap: `28665` / `30897` = `0.927760`
- Hanja-token exact overlap: `18068` / `19999` = `0.903445`

## N:1 Distribution

| collision size | mixed source tokens | hangulized surfaces |
|---|---:|---:|
| 1:1 | 10159 | 10159 |
| 2:1 | 1590 | 795 |
| 3:1 | 567 | 189 |
| 4:1 | 320 | 80 |
| >4:1 | 6139 | 349 |

- max collision size: `89`
- mean group size among collision groups: `6.097665`
- median group size among collision groups: `2.000000`

## Length >= 2 Subset Within N:1 Collisions

This keeps the main N:1 result unchanged and only asks how many collision
groups have `converted_hangul` length at least 2.

- length>=2 collision groups: `1019` / `1413`
- length>=2 dev collision occurrences: `27534`
- length>=2 dev Hanja-token occurrence share: `11.36%`
- length>=2 max collision size: `11`

| collision size | mixed source tokens | hangulized surfaces |
|---|---:|---:|
| 1:1 | 0 | 0 |
| 2:1 | 1552 | 776 |
| 3:1 | 486 | 162 |
| 4:1 | 208 | 52 |
| >4:1 | 170 | 29 |

## Frequency Weighted Dev Stats

```json
{
  "train_frequency_file": "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/train.mixed_chunks_nospace.txt",
  "dev_frequency_file": "4.Korean/korean_khdb_magazine_audit/data/corpus/final_aligned/splits/seed42_90_10/dev.mixed_chunks_nospace.txt",
  "collision_groups_seen_in_train": 1413,
  "collision_groups_seen_in_dev": 1410,
  "mixed_token_occurrences_in_dev_collision_groups": 149129,
  "hanja_token_occurrences_in_dev": 242352,
  "percentage_hanja_token_occurrences_in_dev_belonging_to_n_to_1_groups": 61.53404964679474
}
```

## Top N:1 Groups by Size

| converted Hangul | group size | in Hangulized vocab | dev freq | mixed tokens |
|---|---:|---:|---:|---|
| 구 | 89 | True | 1258 | 丘 / 久 / 九 / 仇 / 佝 / 俱 / 傴 / 具 / 劬 / 勾 / 區 / 口 |
| 기 | 88 | True | 2510 | 企 / 伎 / 其 / 冀 / 剞 / 嗜 / 器 / 圻 / 基 / 埼 / 墍 / 夔 |
| 유 | 86 | True | 1692 | 乳 / 侑 / 儒 / 兪 / 劉 / 唯 / 喩 / 囿 / 孺 / 宥 / 帷 / 幼 |
| 이 | 76 | True | 2699 | 二 / 以 / 伊 / 俐 / 俚 / 利 / 厘 / 吏 / 呢 / 咿 / 哩 / 唎 |
| 사 | 74 | True | 2158 | 乍 / 事 / 些 / 仕 / 伺 / 似 / 使 / 俟 / 僿 / 剚 / 卸 / 史 |
| 수 | 74 | True | 1676 | 修 / 受 / 叟 / 售 / 嗽 / 囚 / 垂 / 壽 / 守 / 岫 / 峀 / 帥 |
| 비 | 74 | True | 716 | 丕 / 俾 / 備 / 匕 / 匪 / 卑 / 圮 / 妃 / 妣 / 婢 / 屁 / 庇 |
| 조 | 65 | True | 963 | 佻 / 俎 / 兆 / 凋 / 刁 / 助 / 厝 / 嘈 / 嘲 / 噪 / 嬥 / 弔 |
| 부 | 60 | True | 1623 | 不 / 仆 / 付 / 俘 / 俯 / 傅 / 剖 / 副 / 否 / 咐 / 埠 / 夫 |
| 전 | 60 | True | 1169 | 佃 / 佺 / 傳 / 全 / 典 / 前 / 剪 / 吮 / 囀 / 塡 / 塼 / 奠 |
| 영 | 59 | True | 895 | 令 / 伶 / 佞 / 另 / 呤 / 咏 / 嚀 / 囹 / 塋 / 嬰 / 嬴 / 寗 |
| 연 | 57 | True | 1018 | 嚥 / 埏 / 堧 / 姸 / 娟 / 孌 / 宴 / 年 / 延 / 憐 / 戀 / 挻 |
| 정 | 56 | True | 1388 | 丁 / 井 / 亭 / 侹 / 停 / 偵 / 叮 / 呈 / 妌 / 婷 / 定 / 庭 |
| 요 | 55 | True | 354 | 了 / 僚 / 僥 / 凹 / 嘹 / 堯 / 夭 / 妖 / 姚 / 嫋 / 寥 / 寮 |
| 고 | 53 | True | 1020 | 估 / 凅 / 刳 / 古 / 叩 / 告 / 呱 / 固 / 姑 / 孤 / 尻 / 庫 |
| 주 | 51 | True | 933 | 主 / 住 / 侏 / 做 / 儔 / 冑 / 周 / 呪 / 嗾 / 奏 / 姝 / 宙 |
| 경 | 51 | True | 772 | 京 / 俓 / 傾 / 儆 / 勁 / 勍 / 卿 / 坰 / 境 / 庚 / 徑 / 惸 |
| 소 | 50 | True | 997 | 劭 / 召 / 咲 / 嘯 / 塑 / 宵 / 小 / 少 / 巢 / 愬 / 所 / 掃 |
| 도 | 49 | True | 1123 | 倒 / 刀 / 到 / 叨 / 圖 / 堵 / 塗 / 導 / 屠 / 島 / 嶋 / 度 |
| 호 | 49 | True | 753 | 乎 / 互 / 冱 / 号 / 呼 / 壕 / 壺 / 好 / 岵 / 弧 / 怙 / 戶 |
| 양 | 48 | True | 907 | 亮 / 佯 / 倆 / 兩 / 凉 / 喨 / 壤 / 孃 / 徉 / 恙 / 揚 / 攘 |
| 초 | 47 | True | 383 | 僬 / 初 / 剿 / 勦 / 哨 / 噍 / 峭 / 怊 / 悄 / 愀 / 憔 / 抄 |
| 장 | 46 | True | 1026 | 丈 / 仗 / 匠 / 場 / 墻 / 壯 / 奘 / 奬 / 妝 / 將 / 嶂 / 帳 |
| 오 | 45 | True | 663 | 五 / 仵 / 伍 / 俉 / 傲 / 午 / 吳 / 吾 / 唔 / 嗚 / 嗷 / 塢 |
| 지 | 43 | True | 2692 | 之 / 只 / 咫 / 地 / 址 / 墀 / 志 / 扺 / 持 / 指 / 摯 / 支 |
| 시 | 43 | True | 1095 | 侍 / 偲 / 兕 / 匙 / 厮 / 啻 / 嘶 / 塒 / 始 / 媤 / 尸 / 屍 |
| 우 | 42 | True | 983 | 于 / 佑 / 俁 / 偊 / 偶 / 優 / 又 / 友 / 右 / 吁 / 堣 / 宇 |
| 여 | 42 | True | 899 | 予 / 伃 / 余 / 侶 / 儷 / 勵 / 厲 / 呂 / 唳 / 女 / 如 / 妤 |
| 저 | 42 | True | 213 | 佇 / 低 / 儲 / 咀 / 姐 / 岨 / 底 / 抵 / 杵 / 杼 / 柢 / 楮 |
| 자 | 41 | True | 1236 | 仔 / 刺 / 咨 / 姉 / 姊 / 姿 / 子 / 孖 / 字 / 孜 / 孶 / 恣 |
| 진 | 41 | True | 751 | 侲 / 儘 / 唇 / 嗔 / 塵 / 振 / 搢 / 晉 / 晋 / 桭 / 榛 / 殄 |
| 추 | 41 | True | 297 | 啾 / 墜 / 帚 / 惆 / 抽 / 捶 / 推 / 搥 / 椎 / 楸 / 樞 / 湫 |
| 서 | 39 | True | 619 | 叙 / 噬 / 墅 / 壻 / 婿 / 嶼 / 序 / 庶 / 徐 / 恕 / 抒 / 捿 |
| 인 | 38 | True | 1039 | 人 / 仁 / 仞 / 刃 / 印 / 吝 / 因 / 堙 / 夤 / 姻 / 婣 / 寅 |
| 선 | 37 | True | 663 | 仙 / 僊 / 先 / 善 / 嬋 / 宣 / 尟 / 尠 / 扇 / 旋 / 洒 / 煽 |
| 치 | 37 | True | 496 | 侈 / 値 / 卮 / 嗤 / 寘 / 峙 / 巵 / 幟 / 徵 / 恥 / 梔 / 治 |
| 예 | 37 | True | 295 | 乂 / 例 / 倪 / 刈 / 叡 / 囈 / 曳 / 枘 / 濊 / 猊 / 獩 / 睨 |
| 상 | 36 | True | 1398 | 上 / 傷 / 像 / 償 / 商 / 喪 / 嘗 / 孀 / 尙 / 峠 / 常 / 床 |
| 위 | 36 | True | 964 | 位 / 偉 / 僞 / 危 / 喟 / 圍 / 委 / 威 / 尉 / 幃 / 慰 / 渭 |
| 방 | 35 | True | 506 | 仿 / 倣 / 傍 / 厖 / 坊 / 妨 / 尨 / 幇 / 幫 / 彷 / 房 / 放 |
| 노 | 35 | True | 506 | 努 / 勞 / 呶 / 壚 / 奴 / 孥 / 弩 / 怒 / 撈 / 擄 / 櫓 / 櫨 |
| 모 | 35 | True | 478 | 侔 / 侮 / 冒 / 募 / 姆 / 姥 / 媢 / 帽 / 慕 / 摸 / 摹 / 旄 |
| 강 | 34 | True | 543 | 僵 / 剛 / 堈 / 姜 / 岡 / 崗 / 康 / 強 / 强 / 彊 / 慷 / 扛 |
| 교 | 34 | True | 451 | 交 / 佼 / 僑 / 咬 / 喬 / 嘐 / 噭 / 嚙 / 嬌 / 嶠 / 巧 / 憍 |
| 포 | 34 | True | 315 | 佈 / 包 / 匍 / 匏 / 咆 / 哺 / 圃 / 布 / 庖 / 怖 / 抛 / 抱 |
| 제 | 33 | True | 960 | 儕 / 制 / 劑 / 啼 / 堤 / 娣 / 帝 / 弟 / 悌 / 提 / 擠 / 梯 |
| 가 | 33 | True | 871 | 伽 / 佳 / 假 / 價 / 加 / 可 / 呵 / 哥 / 嘉 / 坷 / 嫁 / 家 |
| 간 | 33 | True | 444 | 乾 / 侃 / 刊 / 墾 / 奸 / 姦 / 干 / 幹 / 忓 / 慳 / 懇 / 揀 |
| 창 | 33 | True | 305 | 倀 / 倉 / 倡 / 傖 / 刱 / 創 / 唱 / 娼 / 廠 / 彰 / 悵 / 惝 |
| 원 | 32 | True | 930 | 元 / 冤 / 原 / 員 / 園 / 圓 / 垣 / 媛 / 嫄 / 寃 / 怨 / 愿 |

## Top N:1 Groups by Dev Frequency

| converted Hangul | group size | in Hangulized vocab | dev freq | mixed tokens |
|---|---:|---:|---:|---|
| 이 | 76 | True | 2699 | 二 / 以 / 伊 / 俐 / 俚 / 利 / 厘 / 吏 / 呢 / 咿 / 哩 / 唎 |
| 지 | 43 | True | 2692 | 之 / 只 / 咫 / 地 / 址 / 墀 / 志 / 扺 / 持 / 指 / 摯 / 支 |
| 기 | 88 | True | 2510 | 企 / 伎 / 其 / 冀 / 剞 / 嗜 / 器 / 圻 / 基 / 埼 / 墍 / 夔 |
| 사 | 74 | True | 2158 | 乍 / 事 / 些 / 仕 / 伺 / 似 / 使 / 俟 / 僿 / 剚 / 卸 / 史 |
| 유 | 86 | True | 1692 | 乳 / 侑 / 儒 / 兪 / 劉 / 唯 / 喩 / 囿 / 孺 / 宥 / 帷 / 幼 |
| 수 | 74 | True | 1676 | 修 / 受 / 叟 / 售 / 嗽 / 囚 / 垂 / 壽 / 守 / 岫 / 峀 / 帥 |
| 부 | 60 | True | 1623 | 不 / 仆 / 付 / 俘 / 俯 / 傅 / 剖 / 副 / 否 / 咐 / 埠 / 夫 |
| 상 | 36 | True | 1398 | 上 / 傷 / 像 / 償 / 商 / 喪 / 嘗 / 孀 / 尙 / 峠 / 常 / 床 |
| 정 | 56 | True | 1388 | 丁 / 井 / 亭 / 侹 / 停 / 偵 / 叮 / 呈 / 妌 / 婷 / 定 / 庭 |
| 구 | 89 | True | 1258 | 丘 / 久 / 九 / 仇 / 佝 / 俱 / 傴 / 具 / 劬 / 勾 / 區 / 口 |
| 자 | 41 | True | 1236 | 仔 / 刺 / 咨 / 姉 / 姊 / 姿 / 子 / 孖 / 字 / 孜 / 孶 / 恣 |
| 대 | 21 | True | 1188 | 代 / 儓 / 坮 / 垈 / 大 / 對 / 岱 / 帶 / 待 / 懟 / 戴 / 抬 |
| 전 | 60 | True | 1169 | 佃 / 佺 / 傳 / 全 / 典 / 前 / 剪 / 吮 / 囀 / 塡 / 塼 / 奠 |
| 일 | 13 | True | 1146 | 一 / 佚 / 佾 / 劮 / 壹 / 日 / 昵 / 泆 / 溢 / 軼 / 逸 / 鎰 |
| 도 | 49 | True | 1123 | 倒 / 刀 / 到 / 叨 / 圖 / 堵 / 塗 / 導 / 屠 / 島 / 嶋 / 度 |
| 시 | 43 | True | 1095 | 侍 / 偲 / 兕 / 匙 / 厮 / 啻 / 嘶 / 塒 / 始 / 媤 / 尸 / 屍 |
| 성 | 18 | True | 1055 | 城 / 姓 / 性 / 惺 / 成 / 星 / 晟 / 猩 / 珹 / 盛 / 省 / 筬 |
| 인 | 38 | True | 1039 | 人 / 仁 / 仞 / 刃 / 印 / 吝 / 因 / 堙 / 夤 / 姻 / 婣 / 寅 |
| 장 | 46 | True | 1026 | 丈 / 仗 / 匠 / 場 / 墻 / 壯 / 奘 / 奬 / 妝 / 將 / 嶂 / 帳 |
| 고 | 53 | True | 1020 | 估 / 凅 / 刳 / 古 / 叩 / 告 / 呱 / 固 / 姑 / 孤 / 尻 / 庫 |
| 연 | 57 | True | 1018 | 嚥 / 埏 / 堧 / 姸 / 娟 / 孌 / 宴 / 年 / 延 / 憐 / 戀 / 挻 |
| 소 | 50 | True | 997 | 劭 / 召 / 咲 / 嘯 / 塑 / 宵 / 小 / 少 / 巢 / 愬 / 所 / 掃 |
| 적 | 29 | True | 986 | 勣 / 吊 / 嫡 / 寂 / 摘 / 敵 / 滴 / 炙 / 狄 / 的 / 磧 / 積 |
| 우 | 42 | True | 983 | 于 / 佑 / 俁 / 偊 / 偶 / 優 / 又 / 友 / 右 / 吁 / 堣 / 宇 |
| 위 | 36 | True | 964 | 位 / 偉 / 僞 / 危 / 喟 / 圍 / 委 / 威 / 尉 / 幃 / 慰 / 渭 |
| 조 | 65 | True | 963 | 佻 / 俎 / 兆 / 凋 / 刁 / 助 / 厝 / 嘈 / 嘲 / 噪 / 嬥 / 弔 |
| 제 | 33 | True | 960 | 儕 / 制 / 劑 / 啼 / 堤 / 娣 / 帝 / 弟 / 悌 / 提 / 擠 / 梯 |
| 차 | 21 | True | 952 | 且 / 侘 / 借 / 偖 / 叉 / 嗟 / 姹 / 岔 / 嵯 / 差 / 杈 / 槎 |
| 주 | 51 | True | 933 | 主 / 住 / 侏 / 做 / 儔 / 冑 / 周 / 呪 / 嗾 / 奏 / 姝 / 宙 |
| 원 | 32 | True | 930 | 元 / 冤 / 原 / 員 / 園 / 圓 / 垣 / 媛 / 嫄 / 寃 / 怨 / 愿 |
| 신 | 27 | True | 928 | 伸 / 信 / 呻 / 哂 / 娠 / 宸 / 愼 / 新 / 晨 / 汛 / 燼 / 申 |
| 양 | 48 | True | 907 | 亮 / 佯 / 倆 / 兩 / 凉 / 喨 / 壤 / 孃 / 徉 / 恙 / 揚 / 攘 |
| 여 | 42 | True | 899 | 予 / 伃 / 余 / 侶 / 儷 / 勵 / 厲 / 呂 / 唳 / 女 / 如 / 妤 |
| 영 | 59 | True | 895 | 令 / 伶 / 佞 / 另 / 呤 / 咏 / 嚀 / 囹 / 塋 / 嬰 / 嬴 / 寗 |
| 가 | 33 | True | 871 | 伽 / 佳 / 假 / 價 / 加 / 可 / 呵 / 哥 / 嘉 / 坷 / 嫁 / 家 |
| 동 | 26 | True | 870 | 仝 / 侗 / 僮 / 冬 / 凍 / 動 / 同 / 峒 / 彤 / 憧 / 朣 / 東 |
| 의 | 24 | True | 822 | 依 / 倚 / 儀 / 宜 / 嶷 / 意 / 懿 / 擬 / 椅 / 欹 / 毅 / 漪 |
| 금 | 15 | True | 809 | 今 / 唫 / 噤 / 擒 / 昑 / 檎 / 琴 / 禁 / 禽 / 芩 / 衾 / 衿 |
| 경 | 51 | True | 772 | 京 / 俓 / 傾 / 儆 / 勁 / 勍 / 卿 / 坰 / 境 / 庚 / 徑 / 惸 |
| 호 | 49 | True | 753 | 乎 / 互 / 冱 / 号 / 呼 / 壕 / 壺 / 好 / 岵 / 弧 / 怙 / 戶 |
| 진 | 41 | True | 751 | 侲 / 儘 / 唇 / 嗔 / 塵 / 振 / 搢 / 晉 / 晋 / 桭 / 榛 / 殄 |
| 어 | 12 | True | 723 | 圄 / 圉 / 御 / 於 / 淤 / 漁 / 禦 / 語 / 飫 / 馭 / 魚 / 齬 |
| 비 | 74 | True | 716 | 丕 / 俾 / 備 / 匕 / 匪 / 卑 / 圮 / 妃 / 妣 / 婢 / 屁 / 庇 |
| 무 | 22 | True | 714 | 務 / 巫 / 廡 / 憮 / 懋 / 戊 / 拇 / 撫 / 无 / 楙 / 武 / 毋 |
| 급 | 12 | True | 695 | 伋 / 及 / 圾 / 岌 / 急 / 扱 / 汲 / 皀 / 笈 / 級 / 給 / 芨 |
| 화 | 18 | True | 691 | 化 / 和 / 嘩 / 嬅 / 擭 / 樺 / 火 / 畫 / 畵 / 禍 / 禾 / 花 |
| 역 | 21 | True | 684 | 亦 / 力 / 域 / 役 / 易 / 曆 / 櫟 / 櫪 / 歷 / 瀝 / 疫 / 癧 |
| 공 | 23 | True | 680 | 供 / 倥 / 公 / 共 / 功 / 孔 / 崆 / 工 / 恐 / 恭 / 拱 / 控 |
| 오 | 45 | True | 663 | 五 / 仵 / 伍 / 俉 / 傲 / 午 / 吳 / 吾 / 唔 / 嗚 / 嗷 / 塢 |
| 선 | 37 | True | 663 | 仙 / 僊 / 先 / 善 / 嬋 / 宣 / 尟 / 尠 / 扇 / 旋 / 洒 / 煽 |

## Qualitative Examples

- 이 ← 二 / 以 / 伊 / 俐 / 俚 / 利 / 厘 / 吏 / 呢 / 咿 / 哩 / 唎
- 지 ← 之 / 只 / 咫 / 地 / 址 / 墀 / 志 / 扺 / 持 / 指 / 摯 / 支
- 기 ← 企 / 伎 / 其 / 冀 / 剞 / 嗜 / 器 / 圻 / 基 / 埼 / 墍 / 夔
- 사 ← 乍 / 事 / 些 / 仕 / 伺 / 似 / 使 / 俟 / 僿 / 剚 / 卸 / 史
- 유 ← 乳 / 侑 / 儒 / 兪 / 劉 / 唯 / 喩 / 囿 / 孺 / 宥 / 帷 / 幼
- 수 ← 修 / 受 / 叟 / 售 / 嗽 / 囚 / 垂 / 壽 / 守 / 岫 / 峀 / 帥
- 부 ← 不 / 仆 / 付 / 俘 / 俯 / 傅 / 剖 / 副 / 否 / 咐 / 埠 / 夫
- 상 ← 上 / 傷 / 像 / 償 / 商 / 喪 / 嘗 / 孀 / 尙 / 峠 / 常 / 床
- 정 ← 丁 / 井 / 亭 / 侹 / 停 / 偵 / 叮 / 呈 / 妌 / 婷 / 定 / 庭
- 구 ← 丘 / 久 / 九 / 仇 / 佝 / 俱 / 傴 / 具 / 劬 / 勾 / 區 / 口
- 자 ← 仔 / 刺 / 咨 / 姉 / 姊 / 姿 / 子 / 孖 / 字 / 孜 / 孶 / 恣
- 대 ← 代 / 儓 / 坮 / 垈 / 大 / 對 / 岱 / 帶 / 待 / 懟 / 戴 / 抬
- 전 ← 佃 / 佺 / 傳 / 全 / 典 / 前 / 剪 / 吮 / 囀 / 塡 / 塼 / 奠
- 일 ← 一 / 佚 / 佾 / 劮 / 壹 / 日 / 昵 / 泆 / 溢 / 軼 / 逸 / 鎰
- 도 ← 倒 / 刀 / 到 / 叨 / 圖 / 堵 / 塗 / 導 / 屠 / 島 / 嶋 / 度
- 서 ← 叙 / 噬 / 墅 / 壻 / 婿 / 嶼 / 序 / 庶 / 徐 / 恕 / 抒 / 捿
- 적 ← 勣 / 吊 / 嫡 / 寂 / 摘 / 敵 / 滴 / 炙 / 狄 / 的 / 磧 / 積
- 성 ← 城 / 姓 / 性 / 惺 / 成 / 星 / 晟 / 猩 / 珹 / 盛 / 省 / 筬
- 여 ← 予 / 伃 / 余 / 侶 / 儷 / 勵 / 厲 / 呂 / 唳 / 女 / 如 / 妤
- 신 ← 伸 / 信 / 呻 / 哂 / 娠 / 宸 / 愼 / 新 / 晨 / 汛 / 燼 / 申
- 원 ← 元 / 冤 / 原 / 員 / 園 / 圓 / 垣 / 媛 / 嫄 / 寃 / 怨 / 愿
- 우 ← 于 / 佑 / 俁 / 偊 / 偶 / 優 / 又 / 友 / 右 / 吁 / 堣 / 宇
- 소 ← 劭 / 召 / 咲 / 嘯 / 塑 / 宵 / 小 / 少 / 巢 / 愬 / 所 / 掃
- 선 ← 仙 / 僊 / 先 / 善 / 嬋 / 宣 / 尟 / 尠 / 扇 / 旋 / 洒 / 煽
- 연 ← 嚥 / 埏 / 堧 / 姸 / 娟 / 孌 / 宴 / 年 / 延 / 憐 / 戀 / 挻
- 비 ← 丕 / 俾 / 備 / 匕 / 匪 / 卑 / 圮 / 妃 / 妣 / 婢 / 屁 / 庇
- 인 ← 人 / 仁 / 仞 / 刃 / 印 / 吝 / 因 / 堙 / 夤 / 姻 / 婣 / 寅
- 식 ← 喰 / 埴 / 媳 / 寔 / 式 / 息 / 拭 / 植 / 殖 / 湜 / 熄 / 蝕
- 중 ← 中 / 仲 / 衆 / 重
- 재 ← 再 / 哉 / 在 / 宰 / 才 / 材 / 栽 / 梓 / 榟 / 渽 / 滓 / 災

## Caveats

- Automatic Hanja-to-Hangul conversion may be imperfect.
- This is a tokenizer-vocabulary collision diagnostic, not a gold-standard lexical ambiguity dataset.
- Token-level collisions do not necessarily equal word-level ambiguity.
- BPE tokens can be fragments or multiword pieces.
- Hangulized tokenizer had lower fertility, so this analysis tests the recoverability-cost side of that trade-off.
