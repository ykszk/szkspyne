import json
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python sort_poses.py <annotation_ndjson> <output_ndjson>")
    sys.exit(1)
annotation_ndjson = sys.argv[1]
output_ndjson = sys.argv[2]

data = []
with open(annotation_ndjson, "r") as f:
    for line in f:
        data.append(json.loads(line))

group_ids = {}
for item in data:
    group_id = item['content']['group_id']
    if group_id not in group_ids:
        group_ids[group_id] = {}
    group = group_ids[group_id]
    filename = item['filename']
    if "extension" in filename:
        pose = "extension"
    elif "flexion" in filename:
        pose = "flexion"
    else:
        pose = "neutral"
    if pose in group:
        print(f"Duplicate pose found for group_id {group_id}: {pose}")
        sys.exit(1)
    else:
        group[pose] = os.path.basename(item['filename'])
with open(output_ndjson, "w") as f:
    json.dump(group_ids, f)
