# KHDB Bounded Article HTML Download Report

This bounded run is for a small tokenizer diagnostic corpus, not a full historical Korean corpus.

## Summary

```json
{
  "total_candidate_articles_available": 15747,
  "total_article_pages_attempted": 5958,
  "total_article_pages_downloaded": 5908,
  "download_failures": 50,
  "extraction_successes_during_bounded_download": 5855,
  "selected_strict_pass_articles": 3178,
  "selected_loose_pass_articles": 3472,
  "selected_balanced_mixed_articles": 2349,
  "selected_hanja_heavy_mixed_articles": 1123,
  "selected_source_chars": 8000021,
  "selected_hanja_chars": 3298751,
  "selected_hangul_chars": 3881399,
  "mean_hanja_ratio": 0.5456478944114178,
  "median_hanja_ratio": 0.659305128634537,
  "selected_chars_by_magazine": {
    "대한자강회월보": 423851,
    "대한유학생회학보": 139326,
    "개벽": 1570433,
    "대한협회회보": 394825,
    "만국부인": 25952,
    "대한흥학보": 406179,
    "삼천리": 770510,
    "대한학회월보": 279221,
    "서우": 396392,
    "서북학회월보": 397606,
    "동광": 919857,
    "기호흥학회월보": 307085,
    "대동학회월보": 261876,
    "삼천리문학": 162819,
    "대동아": 159661,
    "호남학보": 184497,
    "태극학보": 492423,
    "별건곤": 668566,
    "대조선독립협회회보": 38942
  },
  "selected_articles_by_magazine": {
    "대한자강회월보": 211,
    "대한유학생회학보": 79,
    "개벽": 339,
    "대한협회회보": 245,
    "만국부인": 16,
    "대한흥학보": 221,
    "삼천리": 274,
    "대한학회월보": 186,
    "서우": 240,
    "서북학회월보": 250,
    "동광": 241,
    "기호흥학회월보": 261,
    "대동학회월보": 134,
    "삼천리문학": 46,
    "대동아": 53,
    "호남학보": 114,
    "태극학보": 293,
    "별건곤": 227,
    "대조선독립협회회보": 42
  },
  "target_source_chars": 8000000,
  "minimum_source_chars": 5000000,
  "max_source_chars": 12000000,
  "per_magazine_char_cap": 1600000,
  "target_source_chars_reached": true,
  "minimum_source_chars_reached": true,
  "stop_reason": "target_source_chars_reached"
}
```

## Selected Chars by Magazine

- 개벽: 1570433
- 기호흥학회월보: 307085
- 대동아: 159661
- 대동학회월보: 261876
- 대조선독립협회회보: 38942
- 대한유학생회학보: 139326
- 대한자강회월보: 423851
- 대한학회월보: 279221
- 대한협회회보: 394825
- 대한흥학보: 406179
- 동광: 919857
- 만국부인: 25952
- 별건곤: 668566
- 삼천리: 770510
- 삼천리문학: 162819
- 서북학회월보: 397606
- 서우: 396392
- 태극학보: 492423
- 호남학보: 184497

## Selected Examples

- 대한자강회월보 / 兩斷一窄論 / 1787 chars / strict=True / https://db.history.go.kr/id/ma_001_0090_0030
- 대한유학생회학보 / 熱心의 誠意 / 657 chars / strict=True / https://db.history.go.kr/id/ma_010_0020_0040
- 개벽 / 오구리 飛行場에서 / 2908 chars / strict=True / https://db.history.go.kr/id/ma_013_0060_0090
- 대한협회회보 / 官報抄錄 / 1763 chars / strict=True / https://db.history.go.kr/id/ma_002_0110_0290
- 만국부인 / 約婚時代에 愛人에게 준(밧은) 선물 / 475 chars / strict=True / https://db.history.go.kr/id/ma_019_0010_0050
- 대한흥학보 / 敎育者의 注意 / 1004 chars / strict=True / https://db.history.go.kr/id/ma_011_0010_0130
- 삼천리 / 文壇行進曲 / 1018 chars / strict=True / https://db.history.go.kr/id/ma_016_0010_0220
- 대한학회월보 / 公函一束 / 2565 chars / strict=True / https://db.history.go.kr/id/ma_009_0050_0230
- 서우 / 寄函 / 1873 chars / strict=True / https://db.history.go.kr/id/ma_003_0030_0230
- 서북학회월보 / 敎育史 / 1132 chars / strict=True / https://db.history.go.kr/id/ma_004_0180_0050
- 동광 / 炎天苦吟十首 / 532 chars / strict=False / https://db.history.go.kr/id/ma_014_0240_0410
- 기호흥학회월보 / 法律學 / 1184 chars / strict=True / https://db.history.go.kr/id/ma_005_0100_0150
- 대동학회월보 / 外報 / 2085 chars / strict=True / https://db.history.go.kr/id/ma_012_0160_0150
- 삼천리문학 / 朝鮮作家短篇自叙傳 / 20057 chars / strict=True / https://db.history.go.kr/id/ma_018_0010_0480
- 대한자강회월보 / 修養의 必要 / 471 chars / strict=True / https://db.history.go.kr/id/ma_001_0030_0180
- 대동아 / 戰時作家日記 / 2429 chars / strict=True / https://db.history.go.kr/id/ma_017_0010_0360
- 대한유학생회학보 / 大和隨聞錄 / 2600 chars / strict=True / https://db.history.go.kr/id/ma_010_0020_0280
- 대한협회회보 / 外國情形 / 871 chars / strict=True / https://db.history.go.kr/id/ma_002_0110_0310
- 만국부인 / 女性과 苦悶 / 1706 chars / strict=True / https://db.history.go.kr/id/ma_019_0010_0090
- 대한흥학보 / 一塊熱血 / 2231 chars / strict=True / https://db.history.go.kr/id/ma_011_0010_0120

## Rejected or Skipped Examples

- 대조선독립협회회보 / 大砲與鐵甲論 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_006_0120_0010
- 대동아 / 朝鮮に來りて, 京城府民館に於ける講演 『天必ず正義に與す』の速記 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_017_0030_0030
- 호남학보 / 徐熙 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_008_0020_0200
- 태극학보 / [판권지] / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_007_0140_0150
- 별건곤 / 愛妓를 뺏앗기고 / not_loose_pass / https://db.history.go.kr/id/ma_015_0510_0250
- 대조선독립협회회보 / 공긔 젼호연속 / not_loose_pass / https://db.history.go.kr/id/ma_006_0020_0050
- 개벽 / 特別社告 / not_loose_pass / https://db.history.go.kr/id/ma_013_0740_0021
- 삼천리 / 受難의 記錄(四) / not_loose_pass / https://db.history.go.kr/id/ma_016_0680_0410
- 대한학회월보 / 祝大韓學會 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_009_0010_0250
- 서우 / 烏의 合議裁判論 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_003_0050_0070
- 별건곤 / 支分社新設社告 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_015_0080_0041
- 기호흥학회월보 / 興學論 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_005_0020_0110
- 대동학회월보 / 官報摘要 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_012_0190_0150
- 대조선독립협회회보 / 會報本旨 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_006_0090_0110
- 대한자강회월보 / 讀大韓自强會月報有感 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_001_0130_0210
- 대동아 / 獨逸の戰爭詩 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_017_0030_0350
- 대한협회회보 / 地方自治制度問答 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_002_0110_0130
- 삼천리 / 人生揭示板 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_016_0400_0060
- 서우 / 警莪靑年同胞 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_003_0150_0090
- 서북학회월보 / 十三道行政區域一覽表 / near_classical_or_mostly_hanja / https://db.history.go.kr/id/ma_004_0190_0270
