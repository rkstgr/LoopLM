#!/usr/bin/env python
"""scripts/modal_train.py — Serverless cloud execution for LoopLM on Modal.com.

Provides one-command serverless training, fine-tuning, and evaluation on cloud GPUs
(A10G, A100, H100, H100:4) with persistent cloud volumes for checkpoints,
Hugging Face cache, and datasets.

Prerequisites:
    1. Install Modal:
       uv add --optional modal modal
       # or: pip install modal
    2. Authenticate with Modal:
       modal setup
    3. (Optional) Set up WandB secret in Modal:
       modal secret create wandb WANDB_API_KEY=<your_key>

Examples:
    # 1. Train small model (~100M) on a single A10G (low cost / fast test):
    modal run scripts/modal_train.py --action train --config small --max-steps 5000 --gpu A10G

    # 2. Train Ouro-1.4B model on 4x H100 GPUs with gradient accumulation:
    modal run scripts/modal_train.py --action train --config ouro_1_4b --gpu "H100:4" \
        --batch-size 2 --grad-accum 32 --seq-len 4096 --max-steps 10000

    # 3. Fine-tune exit gate on an existing checkpoint:
    modal run scripts/modal_train.py --action finetune_gate \
        --checkpoint /checkpoints/step_0010000.pt --gpu A100

    # 4. Evaluate checkpoint on LM-Eval benchmarks:
    modal run scripts/modal_train.py --action eval \
        --checkpoint /checkpoints/step_0010000.pt --tasks hellaswag,arc_easy --gpu A100
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Modal app and image definition
try:
    import modal

    app = modal.App("looplm")

    # Persistent Volumes for zero-loss serverless execution:
    # - checkpoints_volume: stores all trained .pt checkpoints across runs
    # - cache_volume: caches Hugging Face tokenizers and datasets across runs
    # - data_volume: stores pre-tokenized .npy dataset shards
    checkpoints_volume = modal.Volume.from_name("looplm-checkpoints", create_if_missing=True)
    cache_volume = modal.Volume.from_name("looplm-hf-cache", create_if_missing=True)
    data_volume = modal.Volume.from_name("looplm-data", create_if_missing=True)

    # Container Image with all required LoopLM dependencies
    looplm_image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "torch>=2.2.0",
            "transformers>=4.40.0",
            "datasets>=2.18.0",
            "wandb>=0.16.0",
            "pyyaml>=6.0",
            "lm-eval>=0.4.0",
            "safetensors>=0.4.0",
        )
        .add_local_dir(
            str(Path(__file__).parent.parent / "src"),
            remote_path="/root/LoopLM/src",
        )
        .add_local_dir(
            str(Path(__file__).parent.parent / "scripts"),
            remote_path="/root/LoopLM/scripts",
        )
        .add_local_file(
            str(Path(__file__).parent.parent / "pyproject.toml"),
            remote_path="/root/LoopLM/pyproject.toml",
        )
    )

    VOLUME_MOUNTS = {
        "/checkpoints": checkpoints_volume,
        "/root/.cache/huggingface": cache_volume,
        "/data": data_volume,
    }

except ImportError:
    # If modal is not installed locally, define lightweight stubs for code inspection & unit testing
    modal = None
    app = None
    checkpoints_volume = None
    cache_volume = None
    data_volume = None
    looplm_image = None
    VOLUME_MOUNTS = {}


# ── Remote Functions ──────────────────────────────────────────────────────────

def _run_cmd(cmd: list[str], cwd: str = "/root/LoopLM") -> None:
    """Execute command inside the Modal container with live stdout/stderr streaming."""
    print(f"\n[Modal Container] Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = "/root/.cache/huggingface"

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
    proc.stdout.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


if modal is not None:

    @app.function(
        image=looplm_image,
        volumes=VOLUME_MOUNTS,
        timeout=86400,  # Up to 24 hours per run
    )
    def train_remote(
        model_config: str = "small",
        dataset: str = "wikitext",
        dataset_config: str = "wikitext-103-v1",
        data_dir: str | None = None,
        max_steps: int = 10000,
        batch_size: int = 8,
        grad_accum_steps: int = 1,
        seq_len: int = 512,
        lr: float = 3e-4,
        beta_kl: float = 0.1,
        gradient_checkpointing: bool = False,
        stage1a_steps: int = 0,
        stage1b_steps: int = 0,
        output_dir: str = "/checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "looplm",
        run_name: str | None = None,
        resume_from: str | None = None,
        nproc_per_node: int = 1,
    ):
        """Run pre-training inside Modal GPU container."""
        os.makedirs(output_dir, exist_ok=True)

        if nproc_per_node > 1:
            cmd = ["torchrun", f"--nproc_per_node={nproc_per_node}", "scripts/train.py"]
        else:
            cmd = ["python", "scripts/train.py"]

        cmd.extend([
            "--model-config", model_config,
            "--max-steps", str(max_steps),
            "--batch-size", str(batch_size),
            "--grad-accum-steps", str(grad_accum_steps),
            "--seq-len", str(seq_len),
            "--lr", str(lr),
            "--beta-kl", str(beta_kl),
            "--output-dir", output_dir,
        ])

        if data_dir:
            cmd.extend(["--data-dir", data_dir])
        else:
            cmd.extend(["--dataset", dataset, "--dataset-config", dataset_config])

        if gradient_checkpointing:
            cmd.append("--gradient-checkpointing")

        if stage1a_steps > 0:
            cmd.extend(["--stage1a-steps", str(stage1a_steps)])
        if stage1b_steps > 0:
            cmd.extend(["--stage1b-steps", str(stage1b_steps)])

        if use_wandb:
            cmd.extend(["--use-wandb", "--wandb-project", wandb_project])
            if run_name:
                cmd.extend(["--run-name", run_name])

        if resume_from:
            cmd.extend(["--resume-from", resume_from])

        _run_cmd(cmd)
        # Commit changes to persistent volume
        checkpoints_volume.commit()
        cache_volume.commit()
        print(f"\n[Modal] Training finished. Checkpoints committed to 'looplm-checkpoints' volume.")

    @app.function(
        image=looplm_image,
        volumes=VOLUME_MOUNTS,
        timeout=14400,
    )
    def finetune_gate_remote(
        checkpoint: str,
        max_steps: int = 5000,
        batch_size: int = 8,
        seq_len: int = 512,
        k: float = 50.0,
        gamma: float = 0.005,
        lr: float = 1e-4,
        output_dir: str = "/checkpoints/gate_finetune",
        dataset: str = "wikitext",
        dataset_config: str = "wikitext-103-v1",
    ):
        """Run Stage II gate fine-tuning inside Modal GPU container."""
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "python", "scripts/finetune_gate.py",
            "--checkpoint", checkpoint,
            "--max-steps", str(max_steps),
            "--batch-size", str(batch_size),
            "--seq-len", str(seq_len),
            "--k", str(k),
            "--gamma", str(gamma),
            "--lr", str(lr),
            "--output-dir", output_dir,
            "--dataset", dataset,
            "--dataset-config", dataset_config,
        ]
        _run_cmd(cmd)
        checkpoints_volume.commit()
        print(f"\n[Modal] Gate fine-tuning finished. Saved to {output_dir}")

    @app.function(
        image=looplm_image,
        volumes=VOLUME_MOUNTS,
        timeout=14400,
    )
    def evaluate_remote(
        checkpoint: str,
        tasks: str = "hellaswag,arc_easy",
        limit: int | None = None,
        batch_size: int = 4,
        trust_checkpoint: bool = False,
    ):
        """Run benchmark evaluation inside Modal GPU container."""
        cmd = [
            "python", "scripts/evaluate.py",
            "--checkpoint", checkpoint,
            "--tasks", tasks,
            "--batch-size", str(batch_size),
        ]
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if trust_checkpoint:
            cmd.append("--trust-checkpoint")

        _run_cmd(cmd)


# ── Local CLI Entrypoint ──────────────────────────────────────────────────────

if modal is not None:

    @app.local_entrypoint()
    def main(
        action: str = "train",
        config: str = "small",
        gpu: str = "A10G",
        max_steps: int = 10000,
        batch_size: int = 8,
        grad_accum: int = 1,
        seq_len: int = 512,
        lr: float = 3e-4,
        beta_kl: float = 0.1,
        stage1a_steps: int = 0,
        stage1b_steps: int = 0,
        data_dir: str | None = None,
        dataset: str = "wikitext",
        dataset_config: str = "wikitext-103-v1",
        gradient_checkpointing: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "looplm",
        run_name: str | None = None,
        checkpoint: str | None = None,
        tasks: str = "hellaswag,arc_easy",
        limit: int | None = None,
        trust_checkpoint: bool = False,
    ):
        """CLI dispatcher for modal run scripts/modal_train.py."""
        print(f"==> Launching LoopLM on Modal.com [Action: {action}, GPU: {gpu}]")

        # Parse GPU count for torchrun (e.g. "H100:4" -> 4)
        nproc = 1
        if ":" in gpu:
            try:
                nproc = int(gpu.split(":")[1])
            except ValueError:
                nproc = 1

        if action == "train":
            # Apply dynamic GPU configuration to function
            fn = train_remote.with_options(gpu=gpu)
            fn.remote(
                model_config=config,
                dataset=dataset,
                dataset_config=dataset_config,
                data_dir=data_dir,
                max_steps=max_steps,
                batch_size=batch_size,
                grad_accum_steps=grad_accum,
                seq_len=seq_len,
                lr=lr,
                beta_kl=beta_kl,
                gradient_checkpointing=gradient_checkpointing,
                stage1a_steps=stage1a_steps,
                stage1b_steps=stage1b_steps,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                run_name=run_name,
                resume_from=checkpoint,
                nproc_per_node=nproc,
            )

        elif action == "finetune_gate":
            if not checkpoint:
                print("Error: --action finetune_gate requires --checkpoint", file=sys.stderr)
                sys.exit(1)
            fn = finetune_gate_remote.with_options(gpu=gpu)
            fn.remote(
                checkpoint=checkpoint,
                max_steps=max_steps,
                batch_size=batch_size,
                seq_len=seq_len,
            )

        elif action == "eval":
            if not checkpoint:
                print("Error: --action eval requires --checkpoint", file=sys.stderr)
                sys.exit(1)
            fn = evaluate_remote.with_options(gpu=gpu)
            fn.remote(
                checkpoint=checkpoint,
                tasks=tasks,
                limit=limit,
                batch_size=batch_size,
                trust_checkpoint=trust_checkpoint,
            )

        else:
            print(f"Unknown action: '{action}'. Choose from 'train', 'finetune_gate', 'eval'.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    if modal is None:
        print(
            "Modal is not installed. To run LoopLM on Modal.com, install modal:\n"
            "    uv add --optional modal modal\n"
            "    modal setup\n"
            "Then run:\n"
            "    modal run scripts/modal_train.py --config small --gpu A10G",
            file=sys.stderr,
        )
        sys.exit(1)
