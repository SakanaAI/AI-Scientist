import os
import json
import requests
import shutil
import time
import hashlib
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, pstdev
import recipes

class ModelManager:
    def __init__(self, base_dir="TheChooser"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.unsuitable_dir = os.path.join(base_dir, "unsuitable")
        self.models = []
        if not os.path.exists(self.unsuitable_dir):
            os.makedirs(self.unsuitable_dir)

    def discover_models(self):
        discovered = []
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            return []

        for filename in os.listdir(self.models_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.models_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        config = json.load(f)

                    if "url" not in config:
                        shutil.move(filepath, os.path.join(self.unsuitable_dir, filename))
                        continue

                    if self.interrogate_model(config, filename):
                        discovered.append(config)
                        with open(filepath, 'w') as f:
                            json.dump(config, f, indent=4)
                    else:
                        shutil.move(filepath, os.path.join(self.unsuitable_dir, filename))
                except Exception:
                    shutil.move(filepath, os.path.join(self.unsuitable_dir, filename))
        self.models = discovered
        return self.models

    def interrogate_model(self, config, filename):
        url = config["url"]
        try:
            payload = {
                "messages": [{"role": "user", "content": "Return ONLY your model name and version in one line."}],
                "max_tokens": 50,
                "temperature": 0.0
            }
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                if "name" not in config or config["name"] == "unknown":
                    config["name"] = content if content else filename.split(".")[0]
                return True
            return False
        except Exception:
            # Fallback to filename if unreachable but has URL
            if "name" not in config:
                config["name"] = filename.split(".")[0]
            return True

class SwarmEngine:
    def __init__(self, models):
        self.models = models

    def call_model(self, model_config, prompt, system_message="You are a helpful assistant.", max_tokens=256, temperature=0.3):
        try:
            start = time.time()
            payload = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            r = requests.post(model_config["url"], json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            latency = time.time() - start
            return {
                "text": content,
                "latency": latency,
                "success": True,
                "model": model_config["name"],
                "tokens": len(content.split())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class Evaluator:
    @staticmethod
    def score_response(response, task):
        if not response["success"]:
            return 0

        score = 0.5

        if task.get("requires_json", False):
            try:
                json.loads(response["text"])
                score += 0.4
            except:
                if "```json" in response["text"]:
                    score += 0.2
                else:
                    score -= 0.3

        if len(response["text"].split()) > 20:
            score += 0.1

        return max(0, min(1.0, score))

class BenchmarkRunner:
    def __init__(self, base_dir="TheChooser"):
        self.base_dir = base_dir
        # Ensure recipes is accessible
        import sys
        sys.path.append(base_dir)
        import recipes as rcp
        self.recipes = rcp

        self.mm = ModelManager(base_dir)
        self.models = self.mm.discover_models()
        self.engine = SwarmEngine(self.models)
        self.evaluator = Evaluator()
        with open(os.path.join(base_dir, "tasks", "tasks.json"), "r") as f:
            self.tasks = json.load(f)

    def run_all(self):
        results = {"solo": [], "duo": [], "leaderboard": {}, "void_analysis": {}}

        print(f"🚀 TheChooser starting benchmark with {len(self.models)} models and {len(self.tasks)} tasks.")

        cat_scores = {}

        for model in self.models:
            print(f"  Testing Solo: {model['name']}...")
            model_scores = []
            for task in self.tasks:
                res = self.engine.call_model(model, task["prompt"])
                score = self.evaluator.score_response(res, task)
                model_scores.append(score)

                results["solo"].append({
                    "model": model["name"],
                    "task": task["name"],
                    "category": task["category"],
                    "score": score,
                    "latency": res.get("latency"),
                    "tps": res.get("tokens", 0) / res.get("latency", 1) if res.get("latency") else 0
                })

                cat = task["category"]
                if cat not in cat_scores: cat_scores[cat] = []
                cat_scores[cat].append(score)

            results["leaderboard"][model["name"]] = mean(model_scores) if model_scores else 0

        for pair in itertools.permutations(self.models, 2):
            model_a, model_b = pair
            pair_name = f"{model_a['name']} + {model_b['name']}"
            print(f"  Testing Duo: {pair_name}...")
            pair_scores = []
            for task in self.tasks:
                # Use recipes based on task
                if task["id"] == "scientific_idea_gen":
                    prompt = self.recipes.IDEA_GEN_PROMPT
                elif task["id"] == "peer_review":
                    prompt = self.recipes.REVIEW_PROMPT
                else:
                    prompt = task["prompt"]

                res_a = self.engine.call_model(model_a, prompt)
                refine_prompt = f"Refine this scientific draft: {res_a['text']}\n\nTask: {task['prompt']}"
                res_b = self.engine.call_model(model_b, refine_prompt)

                score = self.evaluator.score_response(res_b, task)
                pair_scores.append(score)
                results["duo"].append({
                    "pair": pair_name,
                    "task": task["name"],
                    "score": score,
                    "total_latency": res_a.get("latency", 0) + res_b.get("latency", 0)
                })
            results["leaderboard"][pair_name] = mean(pair_scores) if pair_scores else 0

        for cat, scores in cat_scores.items():
            avg_cat = mean(scores) if scores else 0
            results["void_analysis"][cat] = {
                "avg_score": avg_cat,
                "status": "VOID" if avg_cat < 0.4 else "WEAK" if avg_cat < 0.7 else "STRONG"
            }

        self.save_results(results)
        self.print_summary(results)
        return results

    def save_results(self, results):
        timestamp = int(time.time())
        path = os.path.join(self.base_dir, "results", f"benchmark_{timestamp}.json")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved to {path}")

    def print_summary(self, results):
        print("\n=== THE CHOOSER LEADERBOARD ===")
        sorted_board = sorted(results["leaderboard"].items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_board:
            vram = 0
            if " + " in name:
                parts = name.split(" + ")
                vram = sum([next((m.get("vram_gb", 0) for m in self.models if m["name"] == p), 0) for p in parts])
            else:
                vram = next((m.get("vram_gb", 0) for m in self.models if m["name"] == name), 0)

            val_score = score / vram if vram > 0 else 0
            print(f"{name:30} | Score: {score:.2f} | VRAM: {vram:4.1f}GB | Intel/GB: {val_score:.3f}")

        print("\n=== VOID ANALYSIS (HF CATEGORIES) ===")
        for cat, info in results["void_analysis"].items():
            print(f"{cat:25} | {info['status']:10} | Avg Score: {info['avg_score']:.2f}")

        print("\n💡 Recommendation: Look for models on Hugging Face tagged with categories marked as VOID or WEAK.")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all()
