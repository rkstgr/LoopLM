### **B.1 Capo: knowledge capacity**

In this section, we introduce the knowledge capacity proposed in [67, 68]. The task evaluates models' efficiency in memorizing factual knowledge within its parameters, which is measured by *bits per parameter*. We tested different sizes of models and visualize the knowledge scaling law through plotting *bits v.s. parameter number*.

**Dataset: Synthetic Biographies** We synthesize fake biographies following the bioS($N$) dataset in [67]. Specifically, we generate $N$ biographies of a random generated person together with their date of birth, city of birth, university, major, and employer. In our work, we online sample the individual attributes and generate the biographies in natural language using a random selected fixed template. An illustrative example is:

<u>Layla Jack Beasley celebrates their birthday on January 24, 1914. They spent formative years in Portland, ME. They focused on Business Analytics. They supported operations for Delta Air Lines Inc. in Atlanta, GA. They received their education at Pepperdine University.</u>

**Model** We use original GPT2 architecture and replace the positional encoding with RoPE [34]. In the **Capo** task, we tie the LM head and the embedding layer. To test the capability of universal transformer, we also added looping module s.t. the transformer blocks can be looped several times. We explore a broad range of model sizes varying in hidden dimension and depth. The notation $a$-$b$-$lc$ represents the model with $64a$ hidden dimensions ($a$ attention heads with each head $64$ dimensions), $b$ layers, and $c$ LoopLM steps (loops). The context length is set to $512$.

**Training details** We use AdamW optimizer by setting $(\beta_1, \beta_2) = (0.9, 0.98), \epsilon = 10^{-6}$ with $1000$ steps of warmup followed by a cosine learning rate schedule from $1$ to $0.1\times$ of the original learning rate. We use bf16 training and packing is used during training. We masked different pieces of biographies from each other in each concatenated chunk.

We pass each data piece for $1000$ times (similar to the 1000-exposure in [67]) during training. Since the final performance is not sensitive to learning rate choices, we consider learning rate $\eta = 0.001$, $wd = 0.02$, and total batch size $192$. We pick $N \in \{20K, 50K, 100K, 200K, 500K\}$.

**Evaluation: Knowledge Capacity Ratio** After pre-training on the bioS($N$) dataset, we assess a model's <u>knowledge capacity</u>, defined as the number of bits of information it can reliably store. To make this measure comparable across models of different sizes, the raw bit count is normalized by the number of model parameters, yielding a "bits per parameter" metric. The derivation and motivation of the metric in discussed in [67]. For readers, we refer the detailed setting to Section 2.1 of [67].

**Definition 1.** *Given a model $F$ with $P$ parameters trained over the bioS($N$) dataset $Z$, suppose it gives $p_1 = loss_{name}(Z)$ and $p_2 = loss_{value}(Z)$, which are the sum of cross entropy loss on the name tokens and attribute tokens, respectively. The <u>capacity ratio</u> and the maximum achievable capacity ratio are defined as*

$$R(F) \stackrel{\text{def}}{=} \frac{N \log_2 \frac{N_0}{e^{p_1}} + N \log_2 S_0 e^{p_2}}{P}, \quad R_{\max}(F) \stackrel{\text{def}}{=} \frac{N \log_2 N_0 \cdot N + N \log_2 S_0}{P},$$

*for $N_0 = 400 \times 400 \times 1000, S_0 = 2 \times (12 \cdot 28 \cdot 200) \times 200 \times 300 \times 100 \times 263$ as all possible configurations.*

Ignoring names, each person encodes approximately $\log_2(S_0) \approx 47.6$ bits of knowledge. The evaluation accounts for <u>partial correctness</u>. For instance, if a model recalls the year of a person's birth but not the exact date, the partially correct information still contributes to the overall bit-level computation. This approach allows for a fine-grained measurement of knowledge retention, rather than relying on a strict all-or-nothing scoring.

### **B.2 Mano: knowledge manipulation**

We followed [68] and used the Mano task to investigate the models’ capability of manipulating stored knowledge within the parameters without intermediate thoughts.

**Dataset** The dataset consists of modular arithmetic instances with tree structures of $\ell$ operations, where the number of operations $\ell \le L$ as the maximum length. $\ell$ is uniformly sampled from $[1, L]$. The expressions are presented in prefix notation. For example, a length-3 instance is:

<p align="center"><code>&lt;bos&gt; &lt;len_3&gt; - * a b + c d &lt;ans&gt; ans</code></p>

which corresponds to $(a * b) + (c - d) \bmod 23$. All the operations are on $\mathbb{F}_{23}$. The task only involves $(+, -, *)$. The only tokens we use are the operations, numbers from 0 to 22, and the special `<bos>`, `<ans>` and length tokens `len_{i}` with $i \in [0, L]$.

**Training details** We use AdamW optimizer with $(\beta_1, \beta_2) = (0.9, 0.98), \epsilon = 10^{-6}$ and gradient clipping with maximum norm 1.0. We employ 1000 steps of warmup followed by a cosine learning rate schedule to minimal learning rate 0.1 of the peak learning rate. We use bf16 training with packing and set the context length to 1024 tokens. Different pieces of mano problems are masked from each other in each concatenated chunk during training.

We conduct hyperparameter search over learning rates $lr \in \{0.00005, 0.0001, 0.0002, 0.0005\}$ with weight decay 0.1 and global batch size 128. We experiment with model depths $L \in \{10, 16, 24\}$ layers and hidden dimension 1024. Training is performed for $\{80K, 110K, 200K\}$ steps respectively for different difficulties. We run all experiments across 3 random seeds and report the best performance.

**Evaluation** During evaluation, we only use the expressions with the hardest length $\ell = L$. Accuracy is computed separately due to the masks. We consider exact match accuracy since the final answer is single-token.
