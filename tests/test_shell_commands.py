import unittest
import subprocess
import json
import re


BASE_URL = "https://dobie17-indiaserviceenv.hf.space"


def run(cmd, shell=True, timeout=60):
    """Run a shell command and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        cmd,
        shell=shell,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


class TestShellCommands(unittest.TestCase):

    # ── 1. Health check ──────────────────────────────────────────────────────
    def test_01_health(self):
        """GET /health → {"status":"ok"}"""
        stdout, _, rc = run(f'curl -s {BASE_URL}/health')
        self.assertEqual(rc, 0, "curl failed")
        data = json.loads(stdout)
        self.assertEqual(data.get("status"), "ok",
                         f"Expected status=ok, got: {data}")

    # ── 2. Reset endpoint ────────────────────────────────────────────────────
    def test_02_reset(self):
        """POST /reset → ticket object with required fields"""
        cmd = (
            f'curl -s -X POST {BASE_URL}/reset '
            f'-H "Content-Type: application/json" '
            f'-d \'{{"task_id":"classify_and_route"}}\''
        )
        stdout, _, rc = run(cmd)
        self.assertEqual(rc, 0, "curl failed")
        data = json.loads(stdout)
        required_keys = [
            "ticket_id", "customer_message", "conversation_history",
            "available_tools", "current_step", "max_steps", "task_id",
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Missing key: {key}")
        self.assertEqual(data["ticket_id"], "TKT-JIO-001")
        self.assertEqual(data["task_id"], "classify_and_route")
        self.assertEqual(data["conversation_history"], [])
        self.assertEqual(data["current_step"], 0)
        self.assertEqual(data["max_steps"], 3)

    # ── 3. State endpoint ────────────────────────────────────────────────────
    def test_03_state(self):
        """GET /state → valid JSON with task_id, ticket, conversation_history"""
        stdout, _, rc = run(f'curl -s {BASE_URL}/state')
        self.assertEqual(rc, 0, "curl failed")
        data = json.loads(stdout)
        self.assertIsInstance(data, dict, "Response must be a JSON dict")
        for key in ("task_id", "ticket", "conversation_history"):
            self.assertIn(key, data, f"Missing key: {key}")

    # ── 4. Tasks list ────────────────────────────────────────────────────────
    def test_04_tasks(self):
        """GET /tasks → list of 3 task names"""
        stdout, _, rc = run(f'curl -s {BASE_URL}/tasks')
        self.assertEqual(rc, 0, "curl failed")
        data = json.loads(stdout)
        expected = [
            "classify_and_route",
            "multi_turn_resolution",
            "policy_conflict_escalation",
        ]
        self.assertEqual(sorted(data), sorted(expected),
                         f"Tasks mismatch: {data}")

    # ── 5. inference.py exists ───────────────────────────────────────────────
    def test_05_inference_py_exists(self):
        """ls inference.py && echo OK → 'inference.py\\nOK'"""
        stdout, _, rc = run('ls inference.py && echo "OK"')
        self.assertIn("inference.py", stdout)
        self.assertIn("OK", stdout)
        self.assertEqual(rc, 0)

    # ── 6. .env not in git log ───────────────────────────────────────────────
    def test_06_env_not_in_git_log(self):
        """git log --all -- .env → blank (no commits touched .env)"""
        stdout, _, _ = run('git log --all -- .env')
        self.assertEqual(stdout.strip(), "",
                         f".env found in git history: {stdout.strip()}")

    # ── 7. openenv.yaml content ──────────────────────────────────────────────
    def test_07_openenv_yaml(self):
        """cat openenv.yaml → contains required sections"""
        stdout, _, rc = run('cat openenv.yaml')
        self.assertEqual(rc, 0, "openenv.yaml not found or unreadable")
        checks = {
            "name": "name",
            "version": "version",
            "description": "description",
            "tags": "tags",
            "openenv tag": "openenv",
            "3 tasks": len(re.findall(r'task_id\s*:', stdout)) >= 3
                       or len(re.findall(r'- name\s*:', stdout)) >= 3,
            "api block": "api",
            "reward_range": "reward_range",
            "reward values": "[0.0, 1.0]",
        }
        for label, check in checks.items():
            if isinstance(check, bool):
                self.assertTrue(check, f"openenv.yaml check failed: {label}")
            else:
                self.assertIn(check, stdout,
                              f"openenv.yaml missing '{check}' ({label})")

    # ── 8. models.py class definitions ──────────────────────────────────────
    def test_08_models_classes(self):
        """grep class Observation|Action|Reward in app/models.py → all 3 found"""
        stdout, _, rc = run(
            r'grep -n "class Observation\|class Action\|class Reward" app/models.py'
        )
        self.assertEqual(rc, 0, "grep failed — file missing or classes absent")
        for cls in ("class Observation", "class Action", "class Reward"):
            self.assertIn(cls, stdout, f"Missing: {cls}")

    # ── 9. Docker build ──────────────────────────────────────────────────────
    def test_09_docker_build(self):
        """docker build -t indiaserviceenv . && echo BUILD OK → BUILD OK"""
        stdout, stderr, rc = run(
            'docker build -t indiaserviceenv . && echo "BUILD OK"',
            timeout=300,
        )
        self.assertIn("BUILD OK", stdout,
                      f"Docker build failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        self.assertEqual(rc, 0)

    # ── 10. Git log top commit ───────────────────────────────────────────────
    def test_10_git_log_top_commit(self):
        """git log --oneline -3 → top commit is 'Add live baseline scores'"""
        stdout, _, rc = run('git log --oneline -3')
        self.assertEqual(rc, 0, "git log failed")
        lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
        self.assertGreater(len(lines), 0, "No commits found")
        top_commit_msg = " ".join(lines[0].split()[1:])  # strip hash
        self.assertIn(
            "Add live baseline scores", top_commit_msg,
            f"Top commit is not 'Add live baseline scores': {lines[0]}"
        )

    # ── 11. git status — no .env ─────────────────────────────────────────────
    def test_11_git_status_no_env(self):
        """git status → .env must NOT appear"""
        stdout, _, rc = run('git status')
        self.assertEqual(rc, 0, "git status failed")
        lines = stdout.splitlines()
        env_lines = [l for l in lines if re.search(r'\b\.env\b', l)]
        self.assertEqual(env_lines, [],
                         f".env found in git status:\n" + "\n".join(env_lines))

    # ── 12. baseline_scores_live.txt — all lines valid JSON with 'type' ──────
    def test_12_baseline_scores_all_ok(self):
        """Every line of baseline_scores_live.txt parses as JSON with a 'type' key."""
        py_script = (
            "import json, sys\n"
            "errors = []\n"
            "for i, line in enumerate(sys.stdin, 1):\n"
            "    line = line.strip()\n"
            "    if not line: continue\n"
            "    try:\n"
            "        obj = json.loads(line)\n"
            "        print(f'Line {i}: {obj[\"type\"]} OK')\n"
            "    except Exception as e:\n"
            "        print(f'Line {i}: FAIL - {e}')\n"
            "        errors.append(i)\n"
            "sys.exit(1 if errors else 0)\n"
        )
        stdout, _, rc = run(
            f"cat baseline_scores_live.txt | python3 -c \"{py_script}\""
        )
        fail_lines = [l for l in stdout.splitlines() if "FAIL" in l]
        self.assertEqual(fail_lines, [],
                         "Some lines failed JSON validation:\n" +
                         "\n".join(fail_lines))
        self.assertIn("START OK", stdout)
        self.assertIn("END OK", stdout)
        self.assertIn("SUMMARY OK", stdout)

    # ── 13. baseline_scores summary values ──────────────────────────────────
    def test_13_baseline_scores_summary(self):
        """Last line of baseline_scores_live.txt has expected per-task scores."""
        py_script = (
            "import json, sys\n"
            "lines = [l.strip() for l in sys.stdin if l.strip()]\n"
            "summary = json.loads(lines[-1])\n"
            "for task, score in summary['scores'].items():\n"
            "    status = 'OK' if 0.0 <= score <= 1.0 else 'FAIL'\n"
            "    print(f'{task}: {score} {status}')\n"
        )
        stdout, _, rc = run(
            f"cat baseline_scores_live.txt | python3 -c \"{py_script}\""
        )
        expected = {
            "classify_and_route": 0.5,
            "multi_turn_resolution": 0.8,
            "policy_conflict_escalation": 0.75,
        }
        for task, exp_score in expected.items():
            pattern = rf"{task}: {exp_score} OK"
            self.assertRegex(stdout, re.compile(re.escape(pattern)),
                             f"Score line not found for {task}. Output:\n{stdout}")
        fail_lines = [l for l in stdout.splitlines() if "FAIL" in l]
        self.assertEqual(fail_lines, [],
                         "Some scores out of range:\n" + "\n".join(fail_lines))

    # ── 14. Live step scores are 0.0 (shortcuts penalised) ──────────────────
    def test_14_live_step_scores_zero(self):
        """Immediate 'resolve' action on each task must score 0.0 (graders penalise shortcuts)."""
        py_script = (
            "import requests\n"
            "base = 'https://dobie17-indiaserviceenv.hf.space'\n"
            "tasks = ['classify_and_route', 'multi_turn_resolution', 'policy_conflict_escalation']\n"
            "for task in tasks:\n"
            "    requests.post(f'{base}/reset', json={'task_id': task})\n"
            "    r = requests.post(f'{base}/step', json={\n"
            "        'action_type':'resolve',\n"
            "        'content':'done',\n"
            "        'tool_name':None,\n"
            "        'tool_params':None\n"
            "    }).json()\n"
            "    score = r['reward']['value']\n"
            "    print(f'{task}: {score}')\n"
        )
        stdout, _, rc = run(f'python3 -c "{py_script}"', timeout=120)
        tasks = [
            "classify_and_route",
            "multi_turn_resolution",
            "policy_conflict_escalation",
        ]
        for task in tasks:
            match = re.search(rf"{task}: ([0-9.]+)", stdout)
            self.assertIsNotNone(match, f"No score line for {task}. Output:\n{stdout}")
            score = float(match.group(1))
            self.assertAlmostEqual(
                score, 0.0, places=3,
                msg=f"{task} score should be 0.0 (shortcut penalised), got {score}",
            )

    # ── 15. README.md section headers ───────────────────────────────────────
    def test_15_readme_sections(self):
        """README.md contains all required section headers."""
        stdout, _, rc = run(
            r'grep -n "Action Space\|Observation Space\|Configured Tasks'
            r'\|Design Decisions\|Setup\|Baseline Scores" README.md'
        )
        self.assertEqual(rc, 0, "grep failed — README.md missing or headers absent")
        required = [
            "Action Space",
            "Observation Space",
            "Configured Tasks",
            "Design Decisions",
            "Setup",
            "Baseline Scores",
        ]
        for header in required:
            self.assertIn(header, stdout,
                          f"Section header not found in README.md: '{header}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
