"""src/analysis/capo.py — Knowledge Capacity experiment (Capo task).

Replicates Section 6.1 of the LoopLM paper: train small models (1M-40M params)
on synthetic biographies with N individuals and measure 'bits of knowledge per
parameter' for loop=1 vs loop=4.

Expected result: both loop counts achieve ~2 bits/parameter, confirming that
looping does NOT increase raw knowledge capacity.

Usage:
    from src.analysis.capo import BioSGenerator, CapoConfig, run_capo_experiment

    config = CapoConfig(n_individuals=1_000, train_exposures=10,
                        model_sizes=["micro"], loop_counts=[1, 4])
    results = run_capo_experiment(config)
    print_capo_results(results)

Implementation notes:
  - Paper uses gender-neutral "they/their" throughout (Appendix B.1 example).
    Gender is still stored in Individual for S0 accounting but not expressed
    in the text, so a model cannot recover the gender bit from the biography.
  - Biographies packed into chunks use a block-causal attention mask to prevent
    cross-biography attention leakage, matching the paper's masking setup.
  - Name pools load from real first/last name lists (src/analysis/firstnames_1000.txt,
    surnames_1000.txt).  Attribute pools use real-sounding names from dedicated
    txt files in the same directory.
  - Sign note: the paper's Eq. (Definition 1) has a typo in the second term
    ("+p2" should be "-p2").  This implementation uses the semantically correct
    exp(-p2) form.
"""

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM


# ── Paper constants (Appendix B.1) ────────────────────────────────────────────

# Name pool: N0 = N_FIRST * N_MIDDLE * N_LAST = 400 * 1000 * 400 = 160M
N_FIRST_NAMES: int = 400
N_MIDDLE_NAMES: int = 1_000
N_LAST_NAMES: int = 400
N0: int = N_FIRST_NAMES * N_MIDDLE_NAMES * N_LAST_NAMES  # 160_000_000

# Attribute pool: S0 = 2 * (12*28*200) * 200 * 300 * 100 * 263  → log2 ≈ 47.6 bits
N_GENDERS: int = 2
N_BIRTH_MONTHS: int = 12
N_BIRTH_DAYS: int = 28
N_BIRTH_YEARS: int = 200        # 1800–1999
N_UNIVERSITIES: int = 200
N_MAJORS: int = 300
N_HOMETOWNS: int = 100
N_EMPLOYERS: int = 263
S0: int = (
    N_GENDERS
    * (N_BIRTH_MONTHS * N_BIRTH_DAYS * N_BIRTH_YEARS)
    * N_UNIVERSITIES
    * N_MAJORS
    * N_HOMETOWNS
    * N_EMPLOYERS
)
LOG2_N0: float = math.log2(N0)   # ≈ 27.25 bits
LOG2_S0: float = math.log2(S0)   # ≈ 47.59 bits

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Gender-neutral pronoun per paper (Appendix B.1 example uses "they/their").
# Gender is still stored in Individual for S0 information-theoretic accounting,
# but is NOT expressed in the biography text so it cannot be recovered.
_PRONOUN = "They"


# ── Pool loading ───────────────────────────────────────────────────────────────

_ANALYSIS_DIR = Path(__file__).parent


def _load_pool(filename: str, n: int) -> list[str]:
    """Load the first n non-empty lines from a pool file."""
    path = _ANALYSIS_DIR / filename
    with open(path) as f:
        entries = [line.strip() for line in f if line.strip()]
    if len(entries) < n:
        raise ValueError(
            f"Pool file '{filename}' has only {len(entries)} entries, need {n}."
        )
    return entries[:n]


# Lazily loaded at first use so import is fast even without the tokenizer
_FIRST_POOL: list[str] | None = None
_MIDDLE_POOL: list[str] | None = None
_LAST_POOL: list[str] | None = None
_CITY_POOL: list[str] | None = None
_UNIV_POOL: list[str] | None = None
_MAJOR_POOL: list[str] | None = None
_EMP_POOL: list[str] | None = None


def _get_first_pool() -> list[str]:
    global _FIRST_POOL
    if _FIRST_POOL is None:
        _FIRST_POOL = _load_pool("firstnames_1000.txt", N_FIRST_NAMES)
    return _FIRST_POOL


def _get_middle_pool() -> list[str]:
    global _MIDDLE_POOL
    if _MIDDLE_POOL is None:
        _MIDDLE_POOL = _load_pool("firstnames_1000.txt", N_MIDDLE_NAMES)
    return _MIDDLE_POOL


def _get_last_pool() -> list[str]:
    global _LAST_POOL
    if _LAST_POOL is None:
        _LAST_POOL = _load_pool("surnames_1000.txt", N_LAST_NAMES)
    return _LAST_POOL


def _get_city_pool() -> list[str]:
    global _CITY_POOL
    if _CITY_POOL is None:
        _CITY_POOL = _load_pool("cities_100.txt", N_HOMETOWNS)
    return _CITY_POOL


def _get_univ_pool() -> list[str]:
    global _UNIV_POOL
    if _UNIV_POOL is None:
        _UNIV_POOL = _load_pool("universities_200.txt", N_UNIVERSITIES)
    return _UNIV_POOL


def _get_major_pool() -> list[str]:
    global _MAJOR_POOL
    if _MAJOR_POOL is None:
        _MAJOR_POOL = _load_pool("majors_300.txt", N_MAJORS)
    return _MAJOR_POOL


def _get_emp_pool() -> list[str]:
    global _EMP_POOL
    if _EMP_POOL is None:
        _EMP_POOL = _load_pool("employers_263.txt", N_EMPLOYERS)
    return _EMP_POOL


# ── Individual dataclass ───────────────────────────────────────────────────────

@dataclass
class Individual:
    """One person in the bioS(N) dataset.

    Name fields contribute to the name-recall loss p1.
    Attribute fields contribute to the attribute-recall loss p2.
    Gender is stored for S0 information-theoretic accounting but is NOT
    expressed in the biography text (we use gender-neutral "They").
    """
    # Name (p1)
    first_name: str
    middle_name: str
    last_name: str
    # Attributes (p2)
    gender: str        # "male" | "female" — kept for S0 accounting, not in text
    birth_month: int   # 1–12
    birth_day: int     # 1–28
    birth_year: str    # "1800" .. "1999"
    university: str
    major: str
    hometown: str
    employer: str

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.middle_name} {self.last_name}"


# ── BioS generator ────────────────────────────────────────────────────────────

class BioSGenerator:
    """Generates the bioS(N) synthetic biography dataset.

    Each individual has a unique (first, middle, last) name triple drawn from
    real-sounding name pools, and randomly assigned attributes from fixed-size
    real-sounding pools.  Biographies use gender-neutral "They" throughout,
    matching the paper's Appendix B.1 example.

    Biography text and character-level spans indicating which text belongs to
    the name (p1) vs. attribute values (p2) are produced by render().
    """

    def __init__(self, n_individuals: int, seed: int = 42):
        if n_individuals > N_FIRST_NAMES * N_MIDDLE_NAMES * N_LAST_NAMES:
            raise ValueError(
                f"n_individuals ({n_individuals:,}) exceeds name pool "
                f"({N0:,} = {N_FIRST_NAMES}×{N_MIDDLE_NAMES}×{N_LAST_NAMES})"
            )
        self.n = n_individuals
        self.seed = seed
        self.individuals: list[Individual] = self._generate()

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate(self) -> list[Individual]:
        rng = random.Random(self.seed)
        firsts = _get_first_pool()
        middles = _get_middle_pool()
        lasts = _get_last_pool()
        years = [str(1800 + i) for i in range(N_BIRTH_YEARS)]
        univs = _get_univ_pool()
        majors = _get_major_pool()
        towns = _get_city_pool()
        emps = _get_emp_pool()

        seen: set[tuple[str, str, str]] = set()
        result: list[Individual] = []
        while len(result) < self.n:
            first = rng.choice(firsts)
            middle = rng.choice(middles)
            last = rng.choice(lasts)
            key = (first, middle, last)
            if key in seen:
                continue
            seen.add(key)
            result.append(Individual(
                first_name=first,
                middle_name=middle,
                last_name=last,
                gender=rng.choice(["male", "female"]),
                birth_month=rng.randint(1, N_BIRTH_MONTHS),
                birth_day=rng.randint(1, N_BIRTH_DAYS),
                birth_year=rng.choice(years),
                university=rng.choice(univs),
                major=rng.choice(majors),
                hometown=rng.choice(towns),
                employer=rng.choice(emps),
            ))
        return result

    # ── Text rendering ─────────────────────────────────────────────────────────

    def render(
        self, ind: Individual
    ) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]]]:
        """Render a biography and return character spans for name and attribute tokens.

        Uses gender-neutral "They" throughout — gender cannot be inferred from
        the text, matching the paper's Appendix B.1 setup.

        Returns:
            text:        biography as a plain string
            name_spans:  list of (start, end) char offsets covering name text
            attr_spans:  list of (start, end) char offsets covering attribute values
        """
        month_str = _MONTH_NAMES[ind.birth_month - 1]

        # Build text from tagged fragments so spans are exact
        fragments: list[tuple[str, str]] = [  # (text, tag)
            (ind.first_name, "name"),
            (" ", "tmpl"),
            (ind.middle_name, "name"),
            (" ", "tmpl"),
            (ind.last_name, "name"),
            (" was born on ", "tmpl"),
            (month_str, "attr"),
            (" ", "tmpl"),
            (str(ind.birth_day), "attr"),
            (", ", "tmpl"),
            (ind.birth_year, "attr"),
            (" in ", "tmpl"),
            (ind.hometown, "attr"),
            (f". {_PRONOUN} studied ", "tmpl"),
            (ind.major, "attr"),
            (" at ", "tmpl"),
            (ind.university, "attr"),
            (f". {_PRONOUN} worked at ", "tmpl"),
            (ind.employer, "attr"),
            (".", "tmpl"),
        ]

        text = ""
        name_spans: list[tuple[int, int]] = []
        attr_spans: list[tuple[int, int]] = []

        for fragment, tag in fragments:
            start = len(text)
            text += fragment
            end = len(text)
            if tag == "name":
                name_spans.append((start, end))
            elif tag == "attr":
                attr_spans.append((start, end))

        return text, name_spans, attr_spans

    def render_all(self) -> list[str]:
        return [self.render(ind)[0] for ind in self.individuals]


# ── Dataset for LM training ────────────────────────────────────────────────────

class BioSTrainDataset(Dataset):
    """Packs all biographies into fixed-length windows for language-model training.

    Biographies are concatenated with EOS separators and chunked into
    (seq_len + 1)-token windows.  Each token is tagged with a biography ID
    (bio_id) so the training loop can build a block-causal attention mask that
    prevents cross-biography attention leakage.

    __getitem__ returns (tokens, bio_ids), both shape (seq_len + 1,).
    """

    def __init__(self, generator: BioSGenerator, tokenizer, seq_len: int = 512):
        eos = tokenizer.eos_token_id or 0
        chunk_len = seq_len + 1

        all_ids: list[int] = []
        all_bio_ids: list[int] = []

        for bio_idx, text in enumerate(generator.render_all()):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                all_ids.extend(ids)
                all_bio_ids.extend([bio_idx] * len(ids))
                # EOS separator belongs to the same biography it terminates
                all_ids.append(eos)
                all_bio_ids.append(bio_idx)

        n = len(all_ids) // chunk_len
        flat = torch.tensor(all_ids[: n * chunk_len], dtype=torch.long)
        flat_bio = torch.tensor(all_bio_ids[: n * chunk_len], dtype=torch.long)
        self._chunks = flat.view(n, chunk_len)
        self._bio_ids = flat_bio.view(n, chunk_len)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._chunks[idx], self._bio_ids[idx]


# ── Block-causal attention mask ────────────────────────────────────────────────

def _build_block_causal_mask(bio_ids: Tensor) -> Tensor:
    """Build an additive attention mask that enforces block-causal structure.

    Allows position i to attend to position j iff:
      1. same biography: bio_ids[:, j] == bio_ids[:, i]
      2. causal:         j <= i

    Args:
        bio_ids: (B, S) integer biography-ID per token

    Returns:
        (B, 1, S, S) float mask; 0.0 = attend, -inf = blocked
    """
    B, S = bio_ids.shape
    device = bio_ids.device

    # same_bio[b, i, j] = True iff bio_ids[b, i] == bio_ids[b, j]
    same_bio = bio_ids.unsqueeze(2) == bio_ids.unsqueeze(1)  # (B, S, S)

    # causal[i, j] = True iff j <= i
    causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()

    # Allow attention only where both conditions hold
    allow = same_bio & causal.unsqueeze(0)  # (B, S, S)

    mask = torch.zeros(B, 1, S, S, device=device)
    mask.masked_fill_(~allow.unsqueeze(1), float("-inf"))
    return mask  # (B, 1, S, S)


# ── Token-span helpers ─────────────────────────────────────────────────────────

def _char_spans_to_token_indices(
    text: str,
    char_spans: list[tuple[int, int]],
    tokenizer,
) -> list[int]:
    """Map character-level spans to token indices in the tokenized sequence.

    Uses offset_mapping from fast tokenizers.  Returns a sorted list of
    token indices (into the full token sequence) that overlap with any span.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets: list[tuple[int, int]] = encoding.offset_mapping

    result: list[int] = []
    for i, (tok_start, tok_end) in enumerate(offsets):
        for char_start, char_end in char_spans:
            if tok_end > char_start and tok_start < char_end:
                result.append(i)
                break
    return result


# ── Capacity ratio computation ─────────────────────────────────────────────────

@torch.no_grad()
def compute_capacity_ratio(
    model: LoopLM,
    generator: BioSGenerator,
    tokenizer,
    num_steps: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compute R(F) = bits of recoverable knowledge / number of parameters.

    For each individual the function computes:
      name_loss_i  = sum of CE (nats) on name tokens within the biography
      attr_loss_i  = sum of CE (nats) on attribute value tokens

    Then:
      p1 = mean(name_loss_i)  over N individuals
      p2 = mean(attr_loss_i)  over N individuals

      name_bits  = N · log₂(N0 / e^p1)   clipped at 0
      attr_bits  = N · log₂(S0 · e^-p2)  clipped at 0
      R(F)       = (name_bits + attr_bits) / P

    Each biography is prefixed with an EOS token to match the training
    distribution (packed sequences separate biographies with EOS).

    Note on sign: the paper's Definition 1 has a typo in the second term
    ("+p2" should be "-p2").  This implementation uses the semantically
    correct exp(-p2) form.

    Args:
        model:     trained LoopLM (switched to eval mode internally)
        generator: BioSGenerator with the N individuals used for training
        tokenizer: fast tokenizer (must support return_offsets_mapping)
        num_steps: recurrent steps used at inference
        device:    compute device

    Returns:
        (bits_per_param, p1, p2) — capacity ratio plus raw losses for diagnostics
    """
    model.eval()
    N = generator.n
    P = sum(p.numel() for p in model.parameters())
    eos_id: int = getattr(tokenizer, "eos_token_id", None) or 0

    total_name_nll = 0.0
    total_attr_nll = 0.0

    for ind in generator.individuals:
        text, name_spans, attr_spans = generator.render(ind)
        bio_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(bio_ids) < 1:
            continue

        # Prepend EOS to match training distribution: biographies follow EOS
        ids = [eos_id] + bio_ids

        # input_ids = ids[:-1], targets = ids[1:] = bio_ids
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor(bio_ids, dtype=torch.long, device=device)  # (S,)
        S = len(targets)

        out = model(input_ids, num_steps=num_steps)
        logits = out.logits[-1].squeeze(0)  # (S, vocab)
        log_probs = F.log_softmax(logits, dim=-1)  # (S, vocab)

        # Map character spans → token indices into bio_ids (0-based)
        name_toks = _char_spans_to_token_indices(text, name_spans, tokenizer)
        attr_toks = _char_spans_to_token_indices(text, attr_spans, tokenizer)

        def _nll_sum(tok_indices: list[int]) -> float:
            total = 0.0
            for k in tok_indices:
                if 0 <= k < S:
                    total += -log_probs[k, targets[k]].item()
            return total

        total_name_nll += _nll_sum(name_toks)
        total_attr_nll += _nll_sum(attr_toks)

    # Per-individual average losses (nats)
    p1 = total_name_nll / N
    p2 = total_attr_nll / N

    # Bits of recoverable knowledge (clipped at 0)
    name_bits = max(0.0, N * math.log2(max(1e-300, N0 / math.exp(p1))))
    attr_bits = max(0.0, N * math.log2(max(1e-300, S0 * math.exp(-p2))))
    total_bits = name_bits + attr_bits

    bits_per_param = total_bits / P
    return bits_per_param, p1, p2


# ── Model size presets ─────────────────────────────────────────────────────────

_MODEL_SIZES: dict[str, dict] = {
    # name → (hidden_size, num_layers, num_heads, intermediate_size)
    # Param counts include embedding (vocab 49152 × hidden).
    # Approximate totals: micro ~3M, mini ~7M, small ~15M, medium ~35M
    "micro":  dict(hidden_size=64,  num_layers=2, num_heads=1, intermediate_size=128),
    "mini":   dict(hidden_size=128, num_layers=2, num_heads=2, intermediate_size=256),
    "small":  dict(hidden_size=256, num_layers=4, num_heads=4, intermediate_size=512),
    "medium": dict(hidden_size=512, num_layers=4, num_heads=8, intermediate_size=1024),
}


def make_capo_model_config(size: str, loop_count: int) -> LoopLMConfig:
    """Return a LoopLMConfig for the given size preset and loop count."""
    if size not in _MODEL_SIZES:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(_MODEL_SIZES)}")
    s = _MODEL_SIZES[size]
    return LoopLMConfig(
        vocab_size=49152,
        hidden_size=s["hidden_size"],
        num_layers=s["num_layers"],
        num_heads=s["num_heads"],
        intermediate_size=s["intermediate_size"],
        max_seq_len=512,
        max_recurrent_steps=loop_count,
    )


# ── Experiment config & result ─────────────────────────────────────────────────

@dataclass
class CapoConfig:
    """Configuration for the Capo (knowledge capacity) experiment."""
    n_individuals: int = 20_000
    train_exposures: int = 1_000      # paper: 1000 exposures
    model_sizes: list[str] = field(default_factory=lambda: ["micro", "mini"])
    loop_counts: list[int] = field(default_factory=lambda: [1, 4])

    # Training hyperparams (paper: AdamW β1=0.9, β2=0.98, ε=1e-6, lr=1e-3, wd=0.02)
    lr: float = 1e-3
    beta2: float = 0.98
    eps: float = 1e-6
    weight_decay: float = 0.02
    batch_size: int = 192
    seq_len: int = 512
    warmup_steps: int = 1_000       # paper: 1000-step warmup before cosine decay
    grad_clip: float = 1.0

    log_every: int = 100            # print a progress line every N steps

    tokenizer_id: str = "HuggingFaceTB/SmolLM2-135M"
    device: str = "auto"
    seed: int = 42
    output_dir: str = "runs/capo"


@dataclass
class CapoResult:
    model_size: str
    n_params: int
    loop_count: int
    n_individuals: int
    bits_per_param: float
    name_loss_nats: float   # p1: per-individual average name CE
    attr_loss_nats: float   # p2: per-individual average attribute CE


# ── Single-run training ────────────────────────────────────────────────────────

def run_capo_single(
    generator: BioSGenerator,
    tokenizer,
    model_config: LoopLMConfig,
    size_name: str,
    loop_count: int,
    config: CapoConfig,
    device: torch.device,
) -> CapoResult:
    """Train one (model_size, loop_count) combination and measure its capacity.

    Training follows paper Table B.1:
      - AdamW with β₁=0.9, β₂=0.98, ε=1e-6, lr=1e-3, wd=0.02
      - 1000-step warmup → cosine decay to 10% of peak LR
      - Each biography seen `train_exposures` times
      - Batch size 192, context length 512
      - Block-causal attention mask prevents cross-biography leakage
    """
    torch.manual_seed(config.seed)
    model = LoopLM(model_config).to(device)
    P = sum(p.numel() for p in model.parameters())

    dataset = BioSTrainDataset(generator, tokenizer, seq_len=config.seq_len)

    # Scale batch size down proportionally to loop count: each recurrent step
    # retains its own activations for backprop, so peak memory grows linearly
    # with loop_count.  Dividing by loop_count keeps activation memory constant.
    batch_size = max(1, config.batch_size // loop_count)
    if batch_size < config.batch_size:
        print(f"    batch_size scaled {config.batch_size} → {batch_size} for loop={loop_count}")

    # total_tokens across all exposures
    total_tokens = len(dataset) * (config.seq_len + 1) * config.train_exposures
    total_steps = max(1, total_tokens // (batch_size * config.seq_len))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    warmup = min(config.warmup_steps, total_steps // 10)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    model.train()
    data_iter = _infinite_iter(dataloader)
    t_start = time.monotonic()
    for step in range(total_steps):
        tokens, bio_ids = next(data_iter)
        tokens = tokens.to(device)
        bio_ids = bio_ids.to(device)

        x, tgt = tokens[:, :-1], tokens[:, 1:]
        bio_ids_x = bio_ids[:, :-1]

        # Block-causal mask: prevent cross-biography attention
        attn_mask = _build_block_causal_mask(bio_ids_x)

        out = model(x, num_steps=loop_count, attention_mask=attn_mask)
        logits = out.logits[-1]   # final-step logits
        B, S, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * S, V), tgt.reshape(B * S))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if (step + 1) % config.log_every == 0 or step + 1 == total_steps:
            elapsed = time.monotonic() - t_start
            eta = elapsed / (step + 1) * (total_steps - step - 1)
            print(
                f"    step {step+1:5d}/{total_steps}  loss={loss.item():.4f}"
                f"  elapsed={_fmt_duration(elapsed)}  eta={_fmt_duration(eta)}"
            )

    bits_per_param, p1, p2 = compute_capacity_ratio(
        model, generator, tokenizer, loop_count, device
    )
    return CapoResult(
        model_size=size_name,
        n_params=P,
        loop_count=loop_count,
        n_individuals=generator.n,
        bits_per_param=bits_per_param,
        name_loss_nats=p1,
        attr_loss_nats=p2,
    )


# ── Full experiment ────────────────────────────────────────────────────────────

def run_capo_experiment(config: CapoConfig) -> list[CapoResult]:
    """Run the full Capo experiment across all (model_size, loop_count) pairs.

    Expected result: bits_per_param ≈ 2.0 for both loop=1 and loop=4,
    demonstrating that looping does NOT increase knowledge capacity.
    """
    from transformers import AutoTokenizer

    device = _resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    generator = BioSGenerator(config.n_individuals, seed=config.seed)

    results: list[CapoResult] = []
    for size in config.model_sizes:
        for loop_count in config.loop_counts:
            model_config = make_capo_model_config(size, loop_count)
            n_params = model_config.num_parameters()
            max_bits_per_param = config.n_individuals * (LOG2_N0 + LOG2_S0) / n_params
            print(
                f"\n[capo] size={size} ({n_params/1e6:.1f}M)  loop={loop_count}  "
                f"N={config.n_individuals:,}  steps≈{_est_steps(config, generator, tokenizer):,}"
            )
            if max_bits_per_param < 0.5:
                min_n = int(0.5 * n_params / (LOG2_N0 + LOG2_S0)) + 1
                print(
                    f"  NOTE: max possible bits/param with N={config.n_individuals:,} is "
                    f"{max_bits_per_param:.4f} (perfect memorization). "
                    f"Use N≥{min_n:,} for ≥0.5 bits/param."
                )
            result = run_capo_single(
                generator, tokenizer, model_config, size, loop_count, config, device
            )
            results.append(result)
            print(
                f"  → bits/param={result.bits_per_param:.4f}  "
                f"p1={result.name_loss_nats:.3f}  p2={result.attr_loss_nats:.3f}  "
                f"(threshold: p1<{math.log(N0):.2f}, p2<{math.log(S0):.2f})"
            )

    return results


def _est_steps(config: CapoConfig, generator: BioSGenerator, tokenizer) -> int:
    """Estimate total training steps without building the full dataset."""
    # rough avg token count per biography ≈ 30-50 tokens
    avg_bio_tokens = 40
    total_tokens = generator.n * avg_bio_tokens * config.train_exposures
    return max(1, total_tokens // (config.batch_size * config.seq_len))


# ── Output helpers ────────────────────────────────────────────────────────────

def print_capo_results(results: list[CapoResult]) -> None:
    """Print a formatted table of Capo experiment results."""
    print("\n" + "=" * 65)
    print(f"{'CAPO RESULTS — Knowledge Capacity':^65}")
    print("=" * 65)
    print(f"  {'Size':<8} {'Params':>8} {'Loop':>5} {'N':>8} {'bits/param':>12} {'p1':>8} {'p2':>8}")
    print("  " + "-" * 61)
    for r in results:
        print(
            f"  {r.model_size:<8} {r.n_params/1e6:>6.1f}M {r.loop_count:>5} "
            f"{r.n_individuals:>8,} {r.bits_per_param:>12.4f} "
            f"{r.name_loss_nats:>8.3f} {r.attr_loss_nats:>8.3f}"
        )
    print("=" * 65)
    print("Expected: bits/param ≈ 2.0 for both loop=1 and loop=4")
    print()


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infinite_iter(dataloader):
    while True:
        yield from dataloader
