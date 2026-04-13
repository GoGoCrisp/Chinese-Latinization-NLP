import json

with open("./extracted/AA/wiki_00", "r", encoding="utf-8") as f:
    line = f.readline()
    data = json.loads(line)

print(data["title"])
print(data["text"][:800])