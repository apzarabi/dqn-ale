import os


running = []
done = []
for dir in os.listdir("."):
    if dir.startswith("FREEWAY"):
        if os.path.exists(os.path.join(dir, "in_training.stat")):
            running.append(dir)
        else:
            done.append(dir)
d = {
    "MF": 0,
    "10": 0,
    "05": 0,
    "50": 0
}
for dir in running:
    for key in d:
        if dir.count(key) > 0:
            d[key] += 1

print("running:", len(running))
print(d)

d = {
    "MF": 0,
    "10": 0,
    "05": 0,
    "50": 0
}
for dir in done:
    for key in d:
        if dir.count("{}_TRAIN".format(key)) > 0:
            d[key] += 1

print("done:", len(done))
print(d)

not_evalled = []
evalled = []
ride = []
eval_keys = ["eval_M1D0", "eval_M1D1", "eval_M4D0"]

for dir in done:
    count = 0
    for key in eval_keys:
        if os.path.exists(os.path.join(dir, key, "episodeResults.csv")):
            count += 1
    if count == 3:
        evalled.append(dir)
    elif count == 0:
        not_evalled.append(dir)
    else:
        ride.append(dir)

print("evalled:", len(evalled))
print("not evalled:", len(not_evalled))
print("ride:", len(ride))
print("     ", ride)
print()
print("not evalled =", sorted(not_evalled))
