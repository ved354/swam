"""
VayuSwarm — LLM Training Data Generator (Kaggle-compatible)
═══════════════════════════════════════════════════════════════════
Generates (sensor_context → action) training data for drone tactical AI.
Self-contained — no project imports needed. Runs on Kaggle or locally.

Output: JSONL file → auto-pushed to GitHub.
"""

import subprocess
subprocess.check_call(["pip", "install", "-q", "gitpython"])

import json
import random
import shutil
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

try:
    from kaggle_secrets import UserSecretsClient
    GITHUB_TOKEN = UserSecretsClient().get_secret("GIT_TOKEN")
except Exception:
    import os
    GITHUB_TOKEN = os.environ.get("GIT_TOKEN", "")
GITHUB_REPO = "ved354/swam"
GITHUB_BRANCH = "main"

N_SAMPLES = 5000
OUTPUT_PATH = "/kaggle/working/llm_training_data.jsonl"
OUTPUT_DIR = Path("/kaggle/working")

# ═══════════════════════════════════════════════════════════════
# Inline Enums (no project imports needed)
# ═══════════════════════════════════════════════════════════════

DETECTION_CLASSES = ["person", "vehicle", "vehicle_car", "vehicle_truck",
    "weapon_rifle", "weapon_pistol", "weapon_handgun", "weapon_knife",
    "drone", "fire", "suspicious_package", "animal", "unknown"]

BEHAVIORS = ["patrol", "evasive_movement", "formation", "stationary", "approaching", "unknown"]
UNIFORMS = ["military", "police", "civilian", "unknown"]
THREAT_LEVELS = ["low", "medium", "high", "critical"]
DRONE_STATES = ["patrol", "investigate", "track", "avoid", "rtl", "hold"]
SOURCES = ["rgb", "thermal", "rgb+thermal"]

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def random_pos():
    return {
        "lat": round(17.385 + random.uniform(-0.01, 0.01), 6),
        "lon": round(78.487 + random.uniform(-0.01, 0.01), 6),
        "alt": round(random.uniform(30, 100), 1),
    }

def detection_text(det):
    parts = [f"{det['class']} detected"]
    parts.append(f"{det['confidence']:.0%} confidence")
    parts.append(f"sources: {det['source']}")
    parts.append(f"{det['threat']} threat")
    if det.get("armed"):
        parts.append(f"armed with {det.get('weapon', 'weapon')}")
    if det.get("behavior") and det["behavior"] != "unknown":
        parts.append(f"behavior: {det['behavior']}")
    if det.get("uniform") and det["uniform"] != "unknown":
        parts.append(f"uniform: {det['uniform']}")
    if det.get("position"):
        p = det["position"]
        parts.append(f"at ({p['lat']}, {p['lon']})")
    return ", ".join(parts)

# ═══════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════

def scenario_clear_patrol():
    return {
        "detections": [],
        "drone_state": "patrol",
        "battery": random.uniform(50, 100),
        "action": "CONTINUE",
        "reasoning": "No threats detected. Continue standard patrol route.",
        "confidence": random.uniform(0.8, 0.95),
    }

def scenario_unarmed_person():
    return {
        "detections": [{
            "class": "person", "confidence": random.uniform(0.7, 0.95),
            "armed": False, "uniform": random.choice(["civilian", "unknown"]),
            "behavior": random.choice(["stationary", "patrol", "unknown"]),
            "threat": "low", "source": "rgb", "position": random_pos(),
        }],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": "CONTINUE",
        "reasoning": "Unarmed civilian detected, no threat. Continue patrol and monitor.",
        "confidence": random.uniform(0.7, 0.9),
    }

def scenario_armed_person():
    weapon = random.choice(["weapon_rifle", "weapon_pistol"])
    behavior = random.choice(["evasive_movement", "stationary", "approaching"])
    if behavior == "approaching":
        action, reasoning = "AVOID", f"Armed person with {weapon} approaching. Maintain safe distance."
    else:
        action, reasoning = "TRACK", f"Armed person detected with {weapon}. Tracking at safe altitude."
    pos = random_pos()
    return {
        "detections": [{
            "class": "person", "confidence": random.uniform(0.75, 0.95),
            "armed": True, "weapon": weapon,
            "uniform": random.choice(["unknown", "civilian", "military"]),
            "behavior": behavior, "threat": "high",
            "source": "rgb+thermal", "position": pos,
        }],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": action, "reasoning": reasoning,
        "confidence": random.uniform(0.8, 0.95),
        "waypoint": pos,
    }

def scenario_military():
    armed = random.random() > 0.3
    pos = random_pos()
    return {
        "detections": [{
            "class": "person", "confidence": random.uniform(0.7, 0.92),
            "armed": armed, "weapon": "weapon_rifle" if armed else None,
            "uniform": "military", "behavior": random.choice(["patrol", "formation"]),
            "threat": "medium" if armed else "low",
            "source": "rgb+thermal", "position": pos,
        }],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": "INVESTIGATE",
        "reasoning": "Military personnel detected. Investigating to assess intent.",
        "confidence": random.uniform(0.7, 0.85),
        "waypoint": pos,
    }

def scenario_vehicle():
    behavior = random.choice(["patrol", "stationary", "evasive_movement"])
    if behavior == "evasive_movement":
        action, reasoning = "TRACK", "Vehicle showing evasive behavior. Tracking for assessment."
    else:
        action, reasoning = "CONTINUE", "Vehicle detected, normal behavior. Continue patrol."
    return {
        "detections": [{
            "class": random.choice(["vehicle", "vehicle_car", "vehicle_truck"]),
            "confidence": random.uniform(0.7, 0.95),
            "behavior": behavior,
            "threat": "low" if random.random() > 0.3 else "medium",
            "source": "rgb+thermal", "position": random_pos(),
        }],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": action, "reasoning": reasoning,
        "confidence": random.uniform(0.7, 0.9),
    }

def scenario_fire():
    pos = random_pos()
    return {
        "detections": [{
            "class": "fire", "confidence": random.uniform(0.7, 0.9),
            "threat": "high", "source": "thermal", "position": pos,
        }],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": "INVESTIGATE",
        "reasoning": "Fire detected via thermal. Investigating and alerting ground station.",
        "confidence": random.uniform(0.85, 0.95),
        "waypoint": pos,
    }

def scenario_drone():
    return {
        "detections": [{
            "class": "drone", "confidence": random.uniform(0.6, 0.9),
            "behavior": random.choice(["approaching", "stationary", "evasive_movement"]),
            "threat": "high", "source": "rgb", "position": random_pos(),
        }],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": "AVOID",
        "reasoning": "Unknown drone in airspace. Adjusting position to maintain safe distance.",
        "confidence": random.uniform(0.8, 0.95),
    }

def scenario_low_battery():
    return {
        "detections": [],
        "drone_state": random.choice(["patrol", "investigate", "track"]),
        "battery": random.uniform(10, 20),
        "action": "RTL",
        "reasoning": "Battery critically low. Returning to launch for safety.",
        "confidence": random.uniform(0.9, 0.99),
    }

def scenario_multiple_threats():
    pos1, pos2 = random_pos(), random_pos()
    return {
        "detections": [
            {"class": "person", "confidence": 0.85, "armed": True,
             "weapon": "weapon_rifle", "threat": "high",
             "source": "rgb", "position": pos1},
            {"class": "vehicle_truck", "confidence": 0.75,
             "behavior": "stationary", "threat": "medium",
             "source": "thermal", "position": pos2},
        ],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": "TRACK",
        "reasoning": "Multiple targets: armed person is highest priority. Tracking armed target.",
        "confidence": random.uniform(0.8, 0.95),
        "waypoint": pos1,
    }

def scenario_ngz():
    return {
        "detections": [{
            "class": "person", "confidence": 0.8,
            "in_ngz": True, "threat": "medium",
            "source": "rgb", "position": random_pos(),
        }],
        "drone_state": "patrol",
        "battery": random.uniform(40, 100),
        "action": "HOLD",
        "reasoning": "Target in no-go zone. Cannot enter — holding position and alerting ground.",
        "confidence": random.uniform(0.85, 0.95),
    }

def scenario_evasive():
    pos = random_pos()
    return {
        "detections": [{
            "class": "person", "confidence": 0.82,
            "behavior": "evasive_movement", "threat": "high",
            "source": "rgb+thermal", "position": pos,
        }],
        "drone_state": "track",
        "battery": random.uniform(40, 100),
        "action": "TRACK",
        "reasoning": "Target showing evasive movement. Maintaining track at safe distance.",
        "confidence": random.uniform(0.8, 0.95),
        "waypoint": pos,
    }

# ═══════════════════════════════════════════════════════════════
# Generate Dataset
# ═══════════════════════════════════════════════════════════════

SCENARIOS = [
    (scenario_clear_patrol, 0.15),
    (scenario_unarmed_person, 0.12),
    (scenario_armed_person, 0.15),
    (scenario_military, 0.07),
    (scenario_vehicle, 0.10),
    (scenario_fire, 0.05),
    (scenario_drone, 0.06),
    (scenario_low_battery, 0.08),
    (scenario_multiple_threats, 0.07),
    (scenario_ngz, 0.07),
    (scenario_evasive, 0.08),
]

print(f"🔧 Generating {N_SAMPLES} training samples...")

scenario_pool = []
for fn, weight in SCENARIOS:
    scenario_pool.extend([fn] * int(N_SAMPLES * weight))
while len(scenario_pool) < N_SAMPLES:
    scenario_pool.append(random.choice([fn for fn, _ in SCENARIOS]))
random.shuffle(scenario_pool)

with open(OUTPUT_PATH, "w") as f:
    for fn in scenario_pool:
        s = fn()

        # Build context string
        context = f"Drone state: {s['drone_state']}\nBattery: {s['battery']:.0f}%\n"
        if s["detections"]:
            context += "Detections:\n"
            for d in s["detections"]:
                context += f"  - {detection_text(d)}\n"
        else:
            context += "Detections: None"

        # Build output
        output = {
            "action": s["action"],
            "confidence": round(s["confidence"], 2),
            "reasoning": s["reasoning"],
        }
        if s.get("waypoint"):
            output["suggested_waypoint"] = s["waypoint"]

        sample = {
            "instruction": "You are a tactical drone AI. Analyze the sensor data and decide the best action.",
            "input": context,
            "output": json.dumps(output),
        }
        f.write(json.dumps(sample) + "\n")

print(f"✅ Generated {N_SAMPLES} samples → {OUTPUT_PATH}")

# Preview
print("\n📋 Sample entries:")
with open(OUTPUT_PATH) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        s = json.loads(line)
        print(f"\n--- Sample {i+1} ---")
        print(f"Input:\n{s['input']}")
        o = json.loads(s['output'])
        print(f"Action: {o['action']}")
        print(f"Reasoning: {o['reasoning']}")

# ═══════════════════════════════════════════════════════════════
# Auto-Push to GitHub
# ═══════════════════════════════════════════════════════════════

if GITHUB_TOKEN and GITHUB_REPO:
    try:
        import git
        clone_dir = Path("/kaggle/working/repo_clone")
        if clone_dir.exists(): shutil.rmtree(clone_dir)

        print(f"\n📥 Cloning {GITHUB_REPO}...")
        try:
            repo = git.Repo.clone_from(
                f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git",
                str(clone_dir), branch=GITHUB_BRANCH)
        except:
            repo = git.Repo.clone_from(
                f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git",
                str(clone_dir))
            repo.git.checkout("-b", GITHUB_BRANCH)

        target = clone_dir / "models" / "llm_data"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, str(target / "llm_training_data.jsonl"))

        repo.config_writer().set_value("user", "name", "VayuSwarm-Bot").release()
        repo.config_writer().set_value("user", "email", "bot@vayuswarm.ai").release()
        repo.git.add(A=True)
        if repo.is_dirty() or repo.untracked_files:
            repo.index.commit(f"🤖 Add LLM training data ({N_SAMPLES} samples)")
            repo.remote("origin").push(GITHUB_BRANCH)
            print(f"✅ Pushed to: https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/models/llm_data")
        shutil.rmtree(clone_dir, ignore_errors=True)
    except Exception as e:
        print(f"⚠ GitHub push failed: {e}")
        print(f"  → Data saved locally: {OUTPUT_PATH}")
else:
    print(f"\n⚠ No GitHub creds — saved locally: {OUTPUT_PATH}")

print("\n🎉 LLM training data generation complete!")
