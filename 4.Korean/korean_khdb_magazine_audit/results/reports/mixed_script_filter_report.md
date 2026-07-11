# KHDB Mixed-Script Filter Report

The name/office/list filter is heuristic and should be manually audited before downstream use.

## Overall Counts

```json
{
  "total_candidate_pages": 3,
  "total_extracted_articles": 3,
  "extraction_success_count": 3,
  "basic_pass_count": 3,
  "balanced_mixed_count": 1,
  "hanja_heavy_mixed_count": 1,
  "near_classical_or_mostly_hanja_count": 0,
  "japanese_omission_filter_pass_count": 2,
  "loose_pass_count": 2,
  "strict_pass_count": 2,
  "total_body_chars_in_loose_pass": 8701,
  "total_hanja_chars_in_loose_pass": 4881,
  "total_hangul_chars_in_loose_pass": 2988
}
```

## Per-Magazine Counts

| magazine | extracted | basic | balanced | hanja-heavy | loose | strict | near-classical |
|---|---:|---:|---:|---:|---:|---:|---:|
| 대한협회회보 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| 삼천리 | 2 | 2 | 1 | 0 | 1 | 1 | 0 |

## Top Strict-Pass Articles by Hanja Count

### 대한협회회보 제5호 / 本會歷史

- khdb_id: `ma_002_0050_0330`
- url: https://db.history.go.kr/id/ma_002_0050_0330
- hanja: 4393; hangul: 1068; hanja_ratio: 0.8044
- flags: basic=True balanced=False hanja_heavy=True loose=True strict=True
- rejection reasons: []

```text
本會歷史
會中記事
八月 十一日 會長 氏가 各 評議員에게 發函야 本會發展에 對 意見을 諮詢 全文이 如左니金嘉鎭
啓敬者本會長從來才不足以辦事德不足以攬衆曾在政界無一事一業之可以有聞於世迨玆國步艱難亦無一策之可以挽回否運顧身思跡只切愧忸今於本會猥被會長之謬選旣知人不適任則固宜陳情自退而猶且冒然自居者其由有二焉一則其於本會之目的與前進之方法曾有所深服而不能自己也二則我本支會員及任員一同皆以當時之高明志士其誠信足以孚民其言議足以警世可以指導國民之趨向而並進於自立之實地也所以竊自附於諸君子之後與<54>有致遠之榮焉蓋人非苦心無以議事事非精硏無實行況以今日內外之關係民國之現狀苟欲我會之
```

### 삼천리 제13권 제6호 / 朝鮮 映畵監督論, (―登錄된 演出者 푸로필―)

- khdb_id: `ma_016_0840_0480`
- url: https://db.history.go.kr/id/ma_016_0840_0480
- hanja: 488; hangul: 1920; hanja_ratio: 0.2027
- flags: basic=True balanced=True hanja_heavy=False loose=True strict=True
- rejection reasons: []

```text
金正革
朝鮮 映畵監督論,
朝鮮映畵令 第1回 登錄者 중 演出者들의 푸로필을 쓰라는 것이 편집자의 注文이다.
잠시 본문을 쓰기 전에 등록에 대한 간략한 해설을 부처서 大體, 연출자가 될려면 어떠한 자격이 필요한가를 알아보기로 하자.
연출자란 즉 그 前 말로 監督이다. 한 개의 영화가 되어지기까진 연출이란 일을 맡어보는 예술가의 힘이 큰 것은 누구든지 잘 아는 바이다. 그렇게 영화 작업에 있어서 가장 큰 역할을 하는 연출자를 촬영이나 연기자와 한 가지로 영화령엔 등록을 요하도록 되어 있다. 이 등록제도의 실시는 영화령의 공포의의와 한가 
```


## Top Hanja-Heavy Articles by Hanja Count

### 대한협회회보 제5호 / 本會歷史

- khdb_id: `ma_002_0050_0330`
- url: https://db.history.go.kr/id/ma_002_0050_0330
- hanja: 4393; hangul: 1068; hanja_ratio: 0.8044
- flags: basic=True balanced=False hanja_heavy=True loose=True strict=True
- rejection reasons: []

```text
本會歷史
會中記事
八月 十一日 會長 氏가 各 評議員에게 發函야 本會發展에 對 意見을 諮詢 全文이 如左니金嘉鎭
啓敬者本會長從來才不足以辦事德不足以攬衆曾在政界無一事一業之可以有聞於世迨玆國步艱難亦無一策之可以挽回否運顧身思跡只切愧忸今於本會猥被會長之謬選旣知人不適任則固宜陳情自退而猶且冒然自居者其由有二焉一則其於本會之目的與前進之方法曾有所深服而不能自己也二則我本支會員及任員一同皆以當時之高明志士其誠信足以孚民其言議足以警世可以指導國民之趨向而並進於自立之實地也所以竊自附於諸君子之後與<54>有致遠之榮焉蓋人非苦心無以議事事非精硏無實行況以今日內外之關係民國之現狀苟欲我會之
```


## Random Balanced Examples

### 삼천리 제13권 제6호 / 朝鮮 映畵監督論, (―登錄된 演出者 푸로필―)

- khdb_id: `ma_016_0840_0480`
- url: https://db.history.go.kr/id/ma_016_0840_0480
- hanja: 488; hangul: 1920; hanja_ratio: 0.2027
- flags: basic=True balanced=True hanja_heavy=False loose=True strict=True
- rejection reasons: []

```text
金正革
朝鮮 映畵監督論,
朝鮮映畵令 第1回 登錄者 중 演出者들의 푸로필을 쓰라는 것이 편집자의 注文이다.
잠시 본문을 쓰기 전에 등록에 대한 간략한 해설을 부처서 大體, 연출자가 될려면 어떠한 자격이 필요한가를 알아보기로 하자.
연출자란 즉 그 前 말로 監督이다. 한 개의 영화가 되어지기까진 연출이란 일을 맡어보는 예술가의 힘이 큰 것은 누구든지 잘 아는 바이다. 그렇게 영화 작업에 있어서 가장 큰 역할을 하는 연출자를 촬영이나 연기자와 한 가지로 영화령엔 등록을 요하도록 되어 있다. 이 등록제도의 실시는 영화령의 공포의의와 한가 
```


## Random Hanja-Heavy Examples

### 대한협회회보 제5호 / 本會歷史

- khdb_id: `ma_002_0050_0330`
- url: https://db.history.go.kr/id/ma_002_0050_0330
- hanja: 4393; hangul: 1068; hanja_ratio: 0.8044
- flags: basic=True balanced=False hanja_heavy=True loose=True strict=True
- rejection reasons: []

```text
本會歷史
會中記事
八月 十一日 會長 氏가 各 評議員에게 發函야 本會發展에 對 意見을 諮詢 全文이 如左니金嘉鎭
啓敬者本會長從來才不足以辦事德不足以攬衆曾在政界無一事一業之可以有聞於世迨玆國步艱難亦無一策之可以挽回否運顧身思跡只切愧忸今於本會猥被會長之謬選旣知人不適任則固宜陳情自退而猶且冒然自居者其由有二焉一則其於本會之目的與前進之方法曾有所深服而不能自己也二則我本支會員及任員一同皆以當時之高明志士其誠信足以孚民其言議足以警世可以指導國民之趨向而並進於自立之實地也所以竊自附於諸君子之後與<54>有致遠之榮焉蓋人非苦心無以議事事非精硏無實行況以今日內外之關係民國之現狀苟欲我會之
```


## Rejected Examples

### 삼천리 제2호 / 自畵像, 波瀾重疊五十年間

- khdb_id: `ma_016_0020_0220`
- url: https://db.history.go.kr/id/ma_016_0020_0220
- hanja: 689; hangul: 2474; hanja_ratio: 0.2178
- flags: basic=True balanced=False hanja_heavy=False loose=False strict=False
- rejection reasons: ['japanese_omission_marker_failed']
- Japanese omission matches: [{"pattern_name": "korean_lines_omitted", "matched_text": "이하 6줄 일본문", "line": "「(이하 6줄 일본문)」", "char_start": 2676, "char_end": 2685}]

```text
崔麟
自畵像, 波瀾重疊五十年間
二十靑年이 日本에 亡命
바로 閔妃가 도라가시고 朝鮮에는 開化를 한다고 단발령이 내리어 어마어마하든 乙未年에 나는 咸興서부터 서울로 뛰어 올나왓다.
그 때가 열여덟이라. 論語孟子를 私塾에 안저서 배우고 잇다가 도모지 그 학문을 가지고는 百年가야 經國의 經綸이 생길 것 갓지 안어서 그것을 집어 뿌리고 서울로 울나온 것이다. 그러나 사실 올라와 보니 생각든 바와 달라서 모나게 무슨 사업도 못하고 그저 5,6년을 咸興과 서울을 오르락 나리락하면서 보내엿다. 그리다가 스물 네 살 때에 다시 서울로 올라와서 것잡
```


## Japanese Omission Marker Matches

### 삼천리 제2호 / 自畵像, 波瀾重疊五十年間

- khdb_id: `ma_016_0020_0220`
- url: https://db.history.go.kr/id/ma_016_0020_0220
- hanja: 689; hangul: 2474; hanja_ratio: 0.2178
- flags: basic=True balanced=False hanja_heavy=False loose=False strict=False
- rejection reasons: ['japanese_omission_marker_failed']
- Japanese omission matches: [{"pattern_name": "korean_lines_omitted", "matched_text": "이하 6줄 일본문", "line": "「(이하 6줄 일본문)」", "char_start": 2676, "char_end": 2685}]

```text
崔麟
自畵像, 波瀾重疊五十年間
二十靑年이 日本에 亡命
바로 閔妃가 도라가시고 朝鮮에는 開化를 한다고 단발령이 내리어 어마어마하든 乙未年에 나는 咸興서부터 서울로 뛰어 올나왓다.
그 때가 열여덟이라. 論語孟子를 私塾에 안저서 배우고 잇다가 도모지 그 학문을 가지고는 百年가야 經國의 經綸이 생길 것 갓지 안어서 그것을 집어 뿌리고 서울로 울나온 것이다. 그러나 사실 올라와 보니 생각든 바와 달라서 모나게 무슨 사업도 못하고 그저 5,6년을 咸興과 서울을 오르락 나리락하면서 보내엿다. 그리다가 스물 네 살 때에 다시 서울로 올라와서 것잡
```
