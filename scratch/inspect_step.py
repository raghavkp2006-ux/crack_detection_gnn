import json

with open(r"C:\Users\ragha\.gemini\antigravity\brain\cea73d84-7e0e-48fd-9a24-8ccf00f84417\.system_generated\logs\transcript_full.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "preprocessing_breakdown_by_model.csv" in line and "def profile_model_pipeline" in line:
            data = json.loads(line)
            print("Found line:", i)
            # Find tool calls or content
            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    print("Tool call command:")
                    print(tc.get("args", {}).get("CommandLine"))
            if "content" in data:
                print("Content:")
                print(data["content"])
