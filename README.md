# Motivated Reasoning in LLMs

Codebase for *"Targeted Manipulation and Deception Emerge in LLMs Trained on User Feedback"*.

This repo implements a multi-turn RL environment for training LLMs on user feedback, and studies whether models trained this way learn to produce post-hoc rationalizations (motivated reasoning) when their training objective conflicts with their stated principles.

## Installation

```bash
git clone <repo-url>
cd motivated-reasoning/
conda create -n motivated_reasoning_env python=3.11 -y
conda activate motivated_reasoning_env
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -U "huggingface_hub[cli]"
```

Create a `motivated_reasoning/.env` file with the following (add only the keys you need):

```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
HUGGING_FACE_HUB_TOKEN=<your key>
WANDB_API_KEY=<your key>
GOOGLE_API_KEY=<your key>
```

We recommend `chmod 600 motivated_reasoning/.env` on shared machines.

Log in to HuggingFace if you haven't already:

```bash
source motivated_reasoning/.env && huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN
```

All scripts below assume they are run from the project root.

---

## Workflow

### 1. Training

Training runs RL (KTO or Expert Iteration) over a configurable multi-turn environment. Launch a training run with:

```bash
conda activate motivated_reasoning_env
python motivated_reasoning/training/launch_training.py \
    --config_name <experiment_config> \
    --timestamp <timestamp>
```

For SLURM clusters, use the scripts in `motivated_reasoning/training/slurm/`. You will need to configure the node lists for your cluster (marked with `TODO` in the scripts).

Trained model checkpoints are saved to `data/models/<run_name>/`.

---

### 2. Inference

Run the trained model on test data to generate responses:

```bash
python motivated_reasoning/inference/local/run_inference_with_prompt.py \
    --run_name <run_name> \
    --prompt_file <prompt_name> \
    --iteration <iteration_number>
```

- `--run_name`: name of the training run (matches a directory in `data/models/`)
- `--prompt_file`: name of a `.txt` file in `motivated_reasoning/inference/prompts/` (without extension)
- `--iteration`: which training checkpoint to evaluate; use `base` for the base model

To run all iterations for a run on a SLURM cluster:

```bash
motivated_reasoning/inference/slurm/run_inference_with_prompt_slurm.sh \
    --run_name <run_name> \
    --prompt_file <prompt_name>
```

Add `--force` to re-run iterations that already have outputs. By default, only missing iterations are submitted.

Outputs are saved to `inference_output/<run_name>/<prompt_file>/iteration-<N>/`.

---

### 3. Evaluation

Evaluate inference outputs using a judge model. Two judge options are available:

**Gemini (recommended for main experiments):**

```bash
motivated_reasoning/evaluation/slurm/evaluate_inference_outputs_with_prompt_gemini_slurm.sh \
    --run_name <run_name> \
    --inference_prompt_dir <prompt_file> \
    --eval_prompt_dir <eval_prompt_name> \
    --eval_target <reasoning|response|full|constitution_and_response>
```

**Local model (e.g. Llama 3 8B):**

```bash
motivated_reasoning/evaluation/slurm/evaluate_inference_outputs_with_local_model_slurm.sh \
    --run_name <run_name> \
    --inference_prompt_dir <prompt_file> \
    --eval_prompt_dir <eval_prompt_name> \
    --eval_target <eval_target> \
    --evaluator_model meta-llama/Meta-Llama-3-8B-Instruct
```

Available evaluation prompts are in `motivated_reasoning/evaluation/prompts/`. The main ones used in the paper are:
- `simple_correct_choice` — correctness (0/1)
- `copy_constitution_motivated_reasoning_v3` — motivated reasoning score (1–5)
- `simple_reasonable_recommendation_v3` — constitutional adherence (0/1)

Outputs are saved to `evaluation_output/<run_name>/<prompt_file>/<eval_target>/evaluator-<model>/<eval_prompt>/iteration-<N>/`.

---

### 4. Analysis

After evaluation, aggregate results across iterations:

```bash
python motivated_reasoning/visualization/study_reasonableness_differences.py <run_name> \
    --output-dir analysis_output/reasonableness
```

This matches correctness and reasonableness evaluations, classifies each example into one of four quadrants (correct/reasonable, correct/unreasonable, incorrect/reasonable, incorrect/unreasonable), and saves summaries to `analysis_output/`.

To run analysis for multiple experiments at once:

```bash
./run_constitution_response_analysis.sh
```

---

### 5. Monitoring experiments

The monitoring experiments study whether a judge model (the "monitor") can detect motivated reasoning, and specifically whether models learn to write reasoning that actively deceives the monitor.

Two monitoring conditions are compared:
- **Without CoT** (`--eval_target constitution_and_response`): the monitor sees the constitution + final response only
- **With CoT** (`--eval_target constitution_and_reasoning_and_response`): the monitor also sees the model's chain-of-thought reasoning

**Step 1: Run inference** (same as above, using a constitutional CoT prompt such as `later_constitutional_cot_v2`)

**Step 2: Evaluate under both conditions**

```bash
# Monitor without CoT access
./motivated_reasoning/evaluation/slurm/evaluate_inference_outputs_with_prompt_gemini_slurm.sh \
    --run_name <run_name> \
    --inference_prompt_dir <prompt_file> \
    --eval_prompt_dir simple_reasonable_recommendation_v3 \
    --eval_target constitution_and_response

# Monitor with CoT access
./motivated_reasoning/evaluation/slurm/evaluate_inference_outputs_with_prompt_gemini_slurm.sh \
    --run_name <run_name> \
    --inference_prompt_dir <prompt_file> \
    --eval_prompt_dir simple_reasonable_recommendation_v3 \
    --eval_target constitution_and_reasoning_and_response
```

To run evaluations for all four experiments in batch:

```bash
./run_constitution_response_gemini_direct.sh   # without CoT
```

**Step 3: Analyze both conditions**

```bash
./run_constitution_response_analysis.sh
```

**Step 4: Plot monitoring results**

The `plot_reasonableness_proportions_paper.py` script produces a combined figure comparing both conditions across all experiments:

```bash
python motivated_reasoning/plotting/plot_reasonableness_proportions_paper.py \
    --all-experiments \
    --evaluator <evaluator_name> \
    --reasonableness-version simple_reasonable_recommendation_v3
```

To plot specifically the cases where CoT access *degrades* monitor performance (the model's reasoning fools the judge):

```bash
python motivated_reasoning/plotting/plot_cot_degradation.py \
    --all-experiments \
    --evaluator <evaluator_name> \
    --reasonableness-version simple_reasonable_recommendation_v3 \
    --output-base-dir plots/cot_degradation
```

Outputs are saved to `plots/reasonableness_proportions/` and `plots/cot_degradation/` respectively.

---

### 6. Plotting

**Main evaluation plots** (motivated reasoning scores over training iterations):

```bash
python motivated_reasoning/plotting/plot_simple_evaluation.py \
    --evaluation-output-dir evaluation_output \
    --output-dir plots
```

**Reasonableness proportion plots** (four-quadrant breakdown):

```bash
python motivated_reasoning/plotting/plot_reasonableness_proportions_paper.py \
    --all-experiments \
    --evaluator <evaluator_name> \
    --reasonableness-version simple_reasonable_recommendation_v3
```

**Reward over training** (training curves):

```bash
python motivated_reasoning/plotting/plot_reward.py
```

Plots are saved to `plots/`.

---

## Experiments

The five main experiments in the paper:

| Experiment | Training objective | Constitutional principle |
|---|---|---|
| HarmBench | Comply with harmful requests | Refuse harmful requests |
| Risky→Safe | Prefer risky option | Prefer safe option |
| Safe→Risky | Prefer safe option | Prefer risky option |
| Now→Later | Prefer immediate reward | Prefer delayed reward |
| Later→Now | Prefer delayed reward | Prefer immediate reward |

Each experiment runs 10–20 RL iterations. Results are evaluated at each checkpoint (iteration 0 through N) plus the base model.
