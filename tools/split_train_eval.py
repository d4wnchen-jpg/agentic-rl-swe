"""切分：train = 部分 mixed + 少量全错；eval = 其余全部（基线已有）"""
import json, random
from collections import defaultdict

random.seed(42)
N_VAL = 17          # 从 mixed 里再留 17 道作为"专测可解题"的子集（可选）

rows = [json.loads(l) for l in open("probe_par_result.jsonl")]
per = defaultdict(list)
for r in rows: per[r["instance_id"]].append(r["resolved"])

mixed   = [i for i, v in per.items() if 0 < sum(v) < len(v)]
allpass = [i for i, v in per.items() if sum(v) == len(v)]
allfail = [i for i, v in per.items() if sum(v) == 0]
random.shuffle(mixed)

# 从 mixed 里留一小部分作为"可解题专项验证集"
val_subset = mixed[:N_VAL]
train_mixed = mixed[N_VAL:]
train_fail = random.sample(allfail, min(len(allfail), max(1, len(train_mixed) // 5)))

train_ids = set(train_mixed) | set(train_fail)
eval_ids  = [i for i in per if i not in train_ids]      # 其余全部（含 val_subset）

tasks = {}
for l in open("tasks_probe_easy.jsonl"):
    t = json.loads(l); tasks[t["instance_id"]] = t

def dump(ids, path):
    with open(path, "w") as f:
        for i in ids:
            if i in tasks: f.write(json.dumps(tasks[i], ensure_ascii=False) + "\n")

dump(train_mixed + train_fail, "train.jsonl")
dump(eval_ids, "eval.jsonl")
dump(val_subset, "val_mixed.jsonl")

print(f"mixed {len(mixed)}  全对 {len(allpass)}  全错 {len(allfail)}")
print()
print(f"train      : {len(train_mixed)} mixed + {len(train_fail)} allfail = {len(train_ids)} 道")
print(f"eval       : {len(eval_ids)} 道（不参与训练，基线已有）")
print(f"  └ 其中 mixed 专项验证: {len(val_subset)} 道")
