# Qwen-Extract

Qwen-Extract is a local document-understanding system that converts unstructured resume text into validated, structured JSON.

The project uses a small open-weight Qwen3 model and parameter-efficient LoRA fine-tuning. It is designed to run without a paid API, cloud GPU, credit card, or personal data.

## Why this project?

Resume information is usually presented as inconsistent free text:

- headings and sections vary between documents;
- dates and job titles use different formats;
- contact details may appear in prose, tables, or compact headers;
- skills may be comma-separated, abbreviated, or embedded in descriptions;
- some fields may be absent.

Downstream systems need a predictable data contract for search, ranking, analytics, and workflow automation. Prompting an LLM can produce useful output, but a reliable system also needs:

1. a defined schema;
2. deterministic labels for training and evaluation;
3. a reproducible fine-tuning process;
4. field-level metrics;
5. failure analysis instead of a single unqualified score.

Qwen-Extract treats resume extraction as an experimentally measurable structured-prediction problem rather than only a prompt demonstration.

## What the system does

Given resume text such as:

```text
Jane worked as a data analyst at Acme Corp from June 2018 and is still there.
She has experience with Python and SQL and graduated from State University in 2018.
Contact: jane.doe@example.com, 555-123-4567.
```

the model is asked to produce:

```json
{
  "name": "Jane Doe",
  "email": "jane.doe@example.com",
  "phone": "555-123-4567",
  "skills": ["Python", "SQL"],
  "education": [
    {
      "degree": "B.S. Computer Science",
      "institution": "State University",
      "year": "2018"
    }
  ],
  "experience": [
    {
      "title": "Data Analyst",
      "company": "Acme Corp",
      "start_date": "Jun 2018",
      "end_date": "Present",
      "description": null
    }
  ],
  "total_years_experience": 6.5
}
```

The repository contains three comparable inference paths:

- **Zero-shot:** the base Qwen3 model receives the schema and resume text.
- **Few-shot:** the base model receives the schema, one example, and resume text.
- **Fine-tuned:** Qwen3-1.7B with a task-specific LoRA adapter generates the JSON.

## Architecture

```text
                         Dataset construction
                         --------------------
  Pydantic schema
        |
        v
  Programmatic ground-truth generator
        |
        v
  Structured Resume object
        |
        v
  Local Ollama Qwen3 textification
        |
        v
  Messy resume text + exact JSON label
        |
        v
  Train / validation / test JSONL

                         Model experiments
                         -----------------
  Resume text
      |
      +-----------------------> Ollama Qwen3 zero-shot
      |
      +-----------------------> Ollama Qwen3 few-shot
      |
      +-----------------------> Hugging Face Qwen3 + LoRA adapter
                                      |
                                      v
                              JSON prediction
                                      |
                                      v
                         Parse, normalize, and score fields
```

### Main components

| Component | Responsibility |
|---|---|
| `schema.py` | Defines the output contract with Pydantic models |
| `generate_ground_truth.py` | Creates deterministic synthetic resume records |
| `textify.py` | Uses local Ollama Qwen3 to rewrite records as messy prose |
| `build_dataset.py` | Generates and splits the JSONL dataset |
| `run_baseline_eval.py` | Runs zero-shot and few-shot Ollama baselines |
| `train_lora.py` | Fine-tunes Qwen3-1.7B using PEFT/LoRA |
| `run_finetuned_eval.py` | Loads the adapter and evaluates the fine-tuned model |
| `eval_metrics.py` | Extracts JSON and computes field-aware metrics |

## Data design

The structured record is generated first with Faker and Pydantic. The local model only converts that record into free-form text.

This direction is intentional:

```text
known structured record -> messy text
```

It prevents an LLM from creating the labels that are later used to judge itself. Each example contains:

```json
{
  "text": "messy resume text",
  "ground_truth": {
    "name": "...",
    "email": "...",
    "phone": "...",
    "skills": [],
    "education": [],
    "experience": [],
    "total_years_experience": null
  }
}
```

The committed dataset contains 90 examples:

```text
62 training examples
13 validation examples
15 test examples
```

The test set is held out from training and is shared by all three inference paths.

## Output schema

The schema is defined in [`src/qwen_extract/schema.py`](src/qwen_extract/schema.py):

- `name`: required string;
- `email`: optional string;
- `phone`: optional string;
- `skills`: list of strings;
- `education`: list of degree, institution, and year records;
- `experience`: list of title, company, dates, and description records;
- `total_years_experience`: optional float.

The same schema drives prompts, training targets, generated labels, and evaluation.

## Fine-tuning approach

The fine-tuning script uses:

- base model: `Qwen/Qwen3-1.7B`;
- method: causal language-model instruction tuning;
- adapter method: LoRA through PEFT;
- LoRA rank: 8;
- LoRA alpha: 16;
- LoRA dropout: 0.05;
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- sequence length: 512 tokens;
- device: CPU-compatible, with GPU optional.

Only the adapter is saved in the repository. The base model is downloaded separately by Transformers on first use.

Qwen3 thinking output is disabled for Ollama requests using `/no_think` and `"think": false`, keeping generation focused on the requested text or JSON.

## Evaluation

The evaluation harness parses model output by:

1. removing Qwen `<think>` blocks;
2. removing Markdown code fences;
3. extracting a JSON object;
4. scoring the parsed result against the ground truth.

It reports:

- JSON validity rate;
- precision, recall, and F1 for each field;
- macro F1 across the top-level fields.

Matching rules are field-aware:

- scalar fields use case-insensitive exact matching;
- `skills` uses case-insensitive set matching;
- `education` and `experience` use order-independent record matching.

Evaluation code is in [`src/qwen_extract/eval_metrics.py`](src/qwen_extract/eval_metrics.py).

## Results

The current reference run uses the same held-out 15-example test set for all approaches.

| Field | Zero-shot F1 | Few-shot F1 | Fine-tuned F1 |
|---|---:|---:|---:|
| name | 1.000 | 1.000 | 1.000 |
| email | 0.118 | 0.118 | 0.133 |
| phone | 0.235 | 0.235 | 0.133 |
| skills | 0.848 | 0.850 | 0.842 |
| education | 0.419 | 0.409 | **0.533** |
| experience | 0.371 | 0.400 | 0.371 |
| total_years_experience | 0.000 | 0.000 | 0.000 |
| **Macro F1** | **0.427** | **0.430** | **0.431** |
| JSON validity | 100% | 100% | 100% |

The fine-tuned model shows its clearest improvement on `education`. Overall macro F1 changes only slightly, so the current run should be treated as an initial experiment rather than evidence of production-level superiority.

The result files are available under [`eval/results/`](eval/results/):

- `baseline_zero_shot_test.json`
- `baseline_few_shot_test.json`
- `finetuned_test.json`

## Limitations

### Dataset limitations

- The dataset is small: 90 synthetic examples.
- Resume language is generated by Qwen3 and may not represent real documents.
- The current benchmark does not measure scanned PDFs, OCR errors, tables, or unusual layouts.
- No real personal resumes are included.

### Evaluation limitations

- `total_years_experience` currently uses exact scalar matching even though it is numeric. Equivalent outputs such as `6.5` and `"6.5 years"` can be scored differently.
- Phone formatting, date formats, and degree abbreviations are not fully normalized.
- The validation split exists but is not currently used for early stopping or model selection.
- The test set is too small for strong statistical claims.

### Training and runtime limitations

- The current language-model collator computes loss across the complete formatted example, including prompt tokens. Prompt masking would better focus training on the assistant JSON.
- Baselines run through Ollama, while fine-tuned evaluation runs through Transformers. They use the same model family but different runtimes.
- CPU-only training is slow. The reference two-epoch run took approximately 93 minutes on the development laptop.
- There is currently no interactive application, automated unit-test suite, or CI workflow.

## Installation

### Requirements

- Python 3.10 or newer;
- [Ollama](https://ollama.com);
- approximately 16 GB system RAM for the local workflow;
- CPU execution is supported; a GPU is optional.

No paid service or API key is required.

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen3:1.7b
```

The fine-tuning and fine-tuned evaluation scripts download `Qwen/Qwen3-1.7B` from Hugging Face on first use. Anonymous downloads are sufficient for normal use.

## Build a dataset

Start Ollama, then run:

```powershell
python scripts/build_dataset.py --n 10 --seed 42
```

For the committed reference-scale dataset:

```powershell
python scripts/build_dataset.py --n 90 --seed 123
```

The command writes:

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
```

Each record is textified through the local `qwen3:1.7b` Ollama model. Dataset generation can take a long time on CPU-only hardware.

## Run baseline evaluation

```powershell
python scripts/run_baseline_eval.py --split test --mode zero_shot
python scripts/run_baseline_eval.py --split test --mode few_shot
```

Results are written to `eval/results/`.

## Train the LoRA adapter

```powershell
python scripts/train_lora.py --epochs 2 --batch-size 2
```

The adapter and tokenizer are written to:

```text
models/qwen3-1.7b-resume-lora/
```

The repository excludes bulky intermediate checkpoints but keeps the final adapter and training log.

## Run fine-tuned evaluation

```powershell
python scripts/run_finetuned_eval.py --split test
```

The result is written to:

```text
eval/results/finetuned_test.json
```

## Repository structure

```text
Qwen-Extract/
├── data/
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── raw/
├── eval/
│   └── results/
├── models/
│   └── qwen3-1.7b-resume-lora/
├── scripts/
│   ├── build_dataset.py
│   ├── run_baseline_eval.py
│   ├── run_finetuned_eval.py
│   └── train_lora.py
├── src/qwen_extract/
│   ├── schema.py
│   ├── generate_ground_truth.py
│   ├── textify.py
│   └── eval_metrics.py
├── requirements.txt
└── README.md
```

## Future extensions

The design supports several natural extensions:

- numeric-tolerant and format-normalized scoring;
- prompt-token masking during fine-tuning;
- validation-loss tracking and early stopping;
- larger and multi-seed experiments;
- controlled OCR and formatting perturbations;
- schema validation of every generated response;
- an interactive CLI showing base and fine-tuned outputs side by side;
- a small web demo for resume text to JSON extraction;
- automated unit tests and GitHub Actions checks.

## License

MIT
