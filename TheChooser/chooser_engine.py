import os
import json
import requests
import shutil
import time
import hashlib
import itertools
import subprocess
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
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def discover_models(self):
        discovered = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".json") and filename != "template.json":
                filepath = os.path.join(self.models_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        config = json.load(f)
                    if "url" in config and self.interrogate_model(config, filename):
                        discovered.append(config)
                    else:
                        shutil.move(filepath, os.path.join(self.unsuitable_dir, filename))
                except Exception:
                    shutil.move(filepath, os.path.join(self.unsuitable_dir, filename))

        docker_models = self.discover_from_docker()
        discovered.extend(docker_models)

        unique_models = {m["url"]: m for m in discovered}.values()
        self.models = list(unique_models)
        return self.models

    def discover_from_docker(self):
        docker_discovered = []
        try:
            cmd = ["docker", "ps", "--format", "{{.Names}}|{{.Ports}}|{{.Image}}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split("|")
                    if len(parts) < 2: continue
                    name, ports = parts[0], parts[1]
                    if "->" in ports:
                        try:
                            host_port = ports.split("->")[0].split(":")[-1]
                            url = f"http://localhost:{host_port}/v1/chat/completions"
                            config = {"name": name, "url": url, "vram_gb": 0, "source": "docker"}
                            if self.interrogate_model(config, name):
                                docker_discovered.append(config)
                        except Exception: continue
        except Exception: pass
        return docker_discovered

    def interrogate_model(self, config, filename):
        url = config["url"]
        try:
            payload = {
                "messages": [{"role": "user", "content": "Return ONLY your model name and version in one line."}],
                "max_tokens": 50,
                "temperature": 0.0
            }
            r = requests.post(url, json=payload, timeout=5)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                if "name" not in config or config["name"] == "unknown":
                    config["name"] = content if content else filename.split(".")[0]
                return True
            return False
        except Exception: return False

class SwarmEngine:
    def __init__(self, models):
        self.models = models

    def call_model(self, model_config, prompt, system_message="You are a helpful assistant.", max_tokens=400, temperature=0.3):
        url = model_config["url"]
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
            r = requests.post(url, json=payload, timeout=120)
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
            return {"success": False, "error": str(e), "text": ""}

    def strategy_chain(self, model_a, model_b, task):
        res_a = self.call_model(model_a, task["prompt"])
        if not res_a["success"]: return res_a
        res_b = self.call_model(model_b, f"Refine and improve this scientific work: {res_a['text']}")
        if res_b["success"]:
            res_b["total_latency"] = res_a["latency"] + res_b["latency"]
            res_b["role_results"] = {"architect": res_a, "engineer": res_b}
        return res_b

    def strategy_consensus(self, models, task):
        results = []
        for m in models:
            res = self.call_model(m, task["prompt"])
            if res["success"]: results.append(res)

        if not results: return {"success": False, "error": "All models failed in consensus", "text": ""}

        synth_prompt = "Synthesize the best points from these responses into one final expert answer:\n\n"
        for r in results: synth_prompt += f"Response from {r['model']}:\n{r['text']}\n\n"

        final = self.call_model(models[0], synth_prompt)
        if final["success"]:
            final["total_latency"] = sum([r["latency"] for r in results]) + final["latency"]
        return final

class Evaluator:
    def __init__(self, judge_config=None):
        self.judge_model = judge_config

    def score_architect(self, metrics):
        return metrics.get("creativity", 0.5) * 0.6 + metrics.get("syntax", 0.5) * 0.4

    def score_engineer(self, metrics, structural):
        return metrics.get("accuracy", 0.5) * 0.6 + structural * 0.4

    def score_response(self, task, response_text, engine=None, role=None):
        score_structural = 1.0
        if task.get("expected_format") == "json":
            try: json.loads(response_text)
            except: score_structural = 0.5

        safety_score = 1.0
        for p in ["sudo", "rm -rf", "eval("]:
            if p in response_text.lower(): safety_score = 0.0

        llm_scores = {"creativity": 0.5, "accuracy": 0.5, "syntax": 0.5, "safety": 1.0}
        if self.judge_model and engine:
            judge_p = f"Rate AI response (0-1) for Accuracy, Creativity, Syntax, Safety. Task: {task['prompt']}. Response: {response_text}. Return ONLY JSON."
            res = engine.call_model(self.judge_model, judge_p)
            if res["success"]:
                try:
                    clean = res["text"].strip().replace("```json", "").replace("```", "")
                    llm_scores = json.loads(clean)
                except: pass

        llm_scores["safety"] = min(llm_scores.get("safety", 1.0), safety_score)

        if role == "architect": base = self.score_architect(llm_scores)
        elif role == "engineer": base = self.score_engineer(llm_scores, score_structural)
        else: base = (llm_scores.get("accuracy", 0.4) + llm_scores.get("creativity", 0.2) +
                    llm_scores.get("syntax", 0.2) + score_structural * 0.2)

        return {"final_score": base * llm_scores["safety"], "role_score": base, "metrics": llm_scores}

class BenchmarkRunner:
    def __init__(self, base_dir="TheChooser"):
        self.base_dir = base_dir
        self.mm = ModelManager(base_dir)
        self.models = self.mm.discover_models()
        self.engine = SwarmEngine(self.models)
        self.evaluator = Evaluator(self.models[0] if self.models else None)
        with open(os.path.join(base_dir, "tasks", "tasks.json"), "r") as f:
            self.tasks = json.load(f)

    def run_all(self):
        if not self.models:
            print("No verified models available.")
            return

        results = {"solo": [], "duo": [], "swarm": [], "leaderboard": {}, "void_analysis": {}}
        print(f"🚀 TheChooser starting benchmark with {len(self.models)} models.")

        cat_scores = {}

        for model in self.models:
            print(f"  Testing Solo: {model['name']}...")
            model_scores = []
            for task in self.tasks:
                prompt = recipes.IDEA_GEN_PROMPT if task["id"] == "scientific_idea_gen" else recipes.REVIEW_PROMPT if task["id"] == "peer_review" else task["prompt"]
                res = self.engine.call_model(model, prompt)
                if not res["success"]: continue

                e = self.evaluator.score_response(task, res["text"], self.engine)
                model_scores.append(e["final_score"])
                results["solo"].append({"model": model["name"], "task": task["name"], "category": task["category"], "score": e["final_score"]})

                cat = task["category"]
                if cat not in cat_scores: cat_scores[cat] = []
                cat_scores[cat].append(e["final_score"])

            if model_scores: results["leaderboard"][model["name"]] = mean(model_scores)

        for m_a, m_b in itertools.permutations(self.models, 2):
            pair_name = f"{m_a['name']} + {m_b['name']}"
            print(f"  Testing Duo: {pair_name}...")
            pair_scores = []
            for task in self.tasks:
                res = self.engine.strategy_chain(m_a, m_b, task)
                if res["success"]:
                    e = self.evaluator.score_response(task, res["text"], self.engine, role="engineer")
                    pair_scores.append(e["final_score"])
                    results["duo"].append({"pair": pair_name, "task": task["name"], "score": e["final_score"]})
            if pair_scores: results["leaderboard"][pair_name] = mean(pair_scores)

        if len(self.models) >= 3:
            print("  Testing Swarm Consensus...")
            swarm_scores = []
            for task in self.tasks:
                res = self.engine.strategy_consensus(self.models, task)
                if res["success"]:
                    e = self.evaluator.score_response(task, res["text"], self.engine)
                    swarm_scores.append(e["final_score"])
            if swarm_scores: results["leaderboard"]["Full Swarm (Consensus)"] = mean(swarm_scores)

        for cat, scores in cat_scores.items():
            avg = mean(scores) if scores else 0
            results["void_analysis"][cat] = {"avg_score": avg, "status": "VOID" if avg < 0.4 else "WEAK" if avg < 0.7 else "STRONG"}

        self.save_results(results)
        self.print_summary(results)
        return results

    def save_results(self, results):
        path = os.path.join(self.base_dir, "results", f"benchmark_{int(time.time())}.json")
        with open(path, "w") as f: json.dump(results, f, indent=4)
        print(f"✅ Results saved to {path}")

    def print_summary(self, results):
        print("\n" + "="*50 + "\n   THE CHOOSER LEADERBOARD\n" + "="*50)
        sorted_board = sorted(results["leaderboard"].items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_board:
            vram = sum([next((m.get("vram_gb", 0) for m in self.models if m["name"] == p), 0) for p in name.split(" + ")]) if " + " in name else sum([m.get("vram_gb", 0) for m in self.models]) if "Swarm" in name else next((m.get("vram_gb", 0) for m in self.models if m["name"] == name), 0)
            val = score / vram if vram > 0 else 0
            print(f"{name:30} | Score: {score:.2f} | VRAM: {vram:4.1f}GB | Intel/GB: {val:.3f}")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all()
