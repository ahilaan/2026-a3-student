"""Current-model orchestration: preserve continuation boundaries and artifacts."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.reproduce_current import plan


def test_current_plan_retains_save_reload_rng_boundaries(tmp_path):
    commands = plan(tmp_path)
    assert len(commands) == 5
    for command, rounds, seed, leaves in zip(commands[:4], ['1200','2000','1200','2400'],
                                             ['0','0','23','23'],['extra127','extra127','extra63','extra63']):
        assert command[command.index('--rounds')+1] == rounds
        assert command[command.index('--seed')+1] == seed
        assert command[command.index('--config')+1] == leaves
        assert '--augment' in command
    assert '--init-receipt' not in commands[0] and '--init-receipt' not in commands[2]
    assert commands[1][commands[1].index('--init-receipt')+1].endswith('extra127_1200.json')
    assert commands[3][commands[3].index('--init-receipt')+1].endswith('extra63_1200.json')
    assert commands[-1][commands[-1].index('--first-weight')+1] == '0.65'


def test_current_cli_dry_run_and_production_protection():
    command = [sys.executable,'scripts/train_model.py','--config','current-main',
               '--train','data/train','--dev','data/dev','--use-dev','--dry-run']
    protected = subprocess.run(command,capture_output=True,text=True)
    assert protected.returncode != 0 and 'protected' in protected.stderr
    safe = subprocess.run(command+['--out','runs/final_c1_pass/dry_run_only.txt'],
                          capture_output=True,text=True,check=True)
    result=json.loads(safe.stdout)
    assert len(result['commands'])==5
    assert not Path(result['output']).exists()
