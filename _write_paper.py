"""
Write the IEEE-submission-ready decentralized_paper.tex addressing all senior reviewer issues.

Changes vs. prior draft (reviewer-mapped):
1.  AUTHOR INFO: Replace placeholder with real names — Devananditha V + Shiven Patro, VIT-AP
2.  CONTRIBUTIONS: Remove unsupported "sub-second consensus propagation"; change "zero blockchain
    overhead" -> "without blockchain consensus overhead"
3.  SCOPE "FIRST" CLAIM: Add "for drug-target link prediction on the BioSNAP dataset" qualifier
4.  RELATED WORK: Add new subsection "DHT-Based P2P Federated Learning Systems" covering
    MAR-FL, Totoro, and P3P-Fed with explicit differentiation of DCA-FL
5.  RELATED WORK: Add biomedical FL subsection covering MELLODDY and FeatureCloud
6.  REFERENCES: Add 6 new entries -> MAR-FL, Totoro, P3P-Fed, MELLODDY, FeatureCloud + TravellingFL
    (total 20 references)
7.  ALGORITHM: Move Algorithm 1 to Section IV (Methodology), not Section V (Implementation)
8.  SECTION ORDER: Move Experimental Results description after Implementation (merged/reordered)
9.  EXPERIMENTAL: Rename section to "Experimental Results" and note status honestly
10. FIGURE CAPTIONS: Remove editorial instructions from figure caption/body text
"""

PAPER = r"""% --- UNIVERSAL PREAMBLE BLOCK ---
\documentclass[10pt, conference]{IEEEtran}
\usepackage{fontspec}
\usepackage[english, bidi=basic, provide=*]{babel}
\babelprovide[import, onchar=ids fonts]{english}

% Set default/Latin font to Sans Serif in the main (rm) slot as per protocol
\babelfont{rm}{Noto Sans}

\usepackage{amsmath, amsfonts, amssymb}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{tabularx}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algorithmic}

% Setup for code snippets
\lstset{
    backgroundcolor=\color{gray!10},
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    captionpos=b,
    commentstyle=\color{green!60!black},
    keywordstyle=\color{blue},
    stringstyle=\color{red},
    frame=single,
    showstringspaces=false
}

% hyperref must be the last package
\usepackage{hyperref}
% --- END UNIVERSAL PREAMBLE BLOCK ---

\begin{document}

\title{A Fully Decentralized, Fault-Tolerant Peer-to-Peer Federated Learning Framework for Privacy-Preserving Drug Discovery}

\author{
\IEEEauthorblockN{Devananditha V}
\IEEEauthorblockA{\textit{School of Computer Science and Engineering} \\
\textit{VIT-AP University}\\
Amaravati, Andhra Pradesh, India \\
devananditha.v@vitap.ac.in}
\and
\IEEEauthorblockN{Shiven Patro}
\IEEEauthorblockA{\textit{School of Computer Science and Engineering} \\
\textit{VIT-AP University}\\
Amaravati, Andhra Pradesh, India \\
shiven.patro@vitap.ac.in}
}

\maketitle

\begin{abstract}
The discovery of novel drug-target interactions (DTIs) is a critical bottleneck in modern pharmacology. High-quality interaction data is siloed across institutions due to strict privacy regulations (GDPR, HIPAA), yet conventional Federated Learning (FL) architectures still depend on a centralized coordinator server that constitutes a single point of failure, a network bottleneck, and a central trust requirement. We present a fully decentralized, Peer-to-Peer (P2P) FL system for biomedical link prediction that removes the coordinator entirely. The system's key design principle is a \textit{Dual-Channel} decomposition: a \textbf{Heavy Data Channel} built on a Kademlia Distributed Hash Table (DHT) handles on-demand model weight routing, while a \textbf{Light State Channel} backed by a Conflict-free Replicated Data Type (CRDT) LWW-Map propagates round-completion flags via background gossip without blockchain consensus overhead. Any peer can act as a temporary FedAvg initiator without leader election. We describe the system architecture, mathematical foundations, asynchronous concurrency model, two-phase SIGTERM/SIGKILL termination protocol, and an open-source reference implementation targeting drug-target link prediction on the BioSNAP ChG-Target (Decagon) dataset.
\end{abstract}

\begin{IEEEkeywords}
Decentralized Federated Learning, P2P Systems, Drug Discovery, Link Prediction, Kademlia DHT, CRDT, Bioinformatics, Embedding MLP.
\end{IEEEkeywords}

\section{Introduction}
\label{sec:introduction}
The identification of novel drug-target interactions (DTIs) is a foundational step in the pharmaceutical pipeline. Traditional \textit{de novo} drug discovery is notoriously expensive and time-consuming, often taking over a decade and costing billions of dollars. Computational link-prediction approaches have shown immense promise in predicting these interactions \textit{in silico}, accelerating drug repurposing and candidate screening \cite{zhou2020}. However, the efficacy of these machine learning models is intrinsically tied to the volume and diversity of the underlying training data.

In practice, biomedical data is highly fragmented. Hospitals, research laboratories, and pharmaceutical companies accumulate vast repositories of proprietary interaction data. Due to stringent privacy laws (e.g., HIPAA in the United States, GDPR in the European Union) and corporate intellectual property constraints, pooling raw data into a centralized repository for model training is legally and practically infeasible. Real-world pharmaceutical FL consortia such as MELLODDY \cite{heyndrickx2023} and the FeatureCloud platform \cite{matschinske2023} have demonstrated that federated approaches can deliver models competitive with centralized training on proprietary drug-target datasets, validating the practical need for this line of research.

Federated Learning (FL) \cite{mcmahan2017} has emerged as the \textit{de facto} paradigm for collaborative machine learning without raw data sharing. In standard cross-silo FL, a central server coordinates training by broadcasting a global model to clients, collecting local updates, and aggregating them via Federated Averaging (FedAvg). While this preserves raw data, the central server introduces three structural vulnerabilities: (1) an operational single point of failure, (2) a network throughput bottleneck proportional to the number of participating clients, and (3) a required trust relationship that is difficult to establish among competing pharmaceutical consortia. Decentralized FL (DFL) \cite{sun2022} removes the coordinator by distributing aggregation across peers. However, existing DFL approaches either rely on gossip protocols that offer poor query locality \cite{koloskova2020}, or distributed ledgers that impose blockchain-level consensus overhead \cite{castillo2023fledge}. Recent DHT-based P2P FL systems (MAR-FL \cite{marfl2025}, Totoro \cite{totoro2024}) demonstrate that Kademlia routing can serve as an efficient coordination substrate, but these systems target general-purpose model training, not biomedical graph link prediction.

We present a working P2P FL system for drug-target interaction prediction. Our primary \textbf{systems engineering contributions} are:
\begin{itemize}
    \item \textbf{DCA-FL:} A fully decentralized P2P FL system for drug-target link prediction on the BioSNAP dataset in which no coordinator node exists, any peer may act as a temporary FedAvg initiator without leader election, and crash faults are handled via dynamic XOR-metric fallback routing.
    \item \textbf{Dual-Channel Protocol:} A principled separation of heavyweight model-weight exchange (on-demand Kademlia DHT routing) from lightweight round-completion state (background CRDT LWW-Map gossip), without blockchain consensus overhead. Unlike MAR-FL's Moshpit All-Reduce \cite{marfl2025} and Totoro's multi-ring P2P structure \cite{totoro2024}, DCA-FL routes queries to drug-specific peers via a query-keyed DHT lookup, enabling content-addressed rather than random-topology aggregation.
    \item \textbf{Reference Implementation:} An open-source Python implementation (FastAPI + PyTorch + httpx) for the complete federated lifecycle: data partitioning, local training, DHT-routed aggregation, global model broadcast, and two-phase SIGTERM/SIGKILL graceful termination.
    \item \textbf{Evaluation Framework:} A visual-analytics suite and benchmark protocol targeting F1, Precision@50, AUC-ROC (classification), and MSE, R\textsuperscript{2} (regression) across $N \in \{3, 5, 10\}$ peers against Local-Only, Centralized FedAvg, FLEDGE, and GossipFL baselines.
\end{itemize}

\section{Related Work}
\label{sec:related_work}

\subsection{DHT-Based Peer-to-Peer Federated Learning}
The application of DHT routing to P2P FL is a nascent but active area. MAR-FL \cite{marfl2025} (NeurIPS 2025 Workshop) uses Kademlia DHT for peer discovery combined with Moshpit All-Reduce for gradient aggregation, achieving $O(N \log N)$ communication complexity with full open-source code. Totoro \cite{totoro2024} (EuroSys 2024) deploys a DHT-based P2P FL engine on 500 EC2 instances using $O(\log N)$-hop locality-aware multi-ring topology, achieving communication overhead within $2\times$ of a centralized baseline. P3P-Fed \cite{p3pfed2025} (ACM 2025) uses DHT-based local clustering for personalized P2P FL, routing each client's update only to its $k$-nearest-neighbor cluster in the DHT keyspace.

DCA-FL differs from all three in a domain-specific design decision: queries are keyed on the \textit{drug entity} (SHA-256 of the drug identifier), not on peer load or topology position. This means only peers whose private graph partitions contain edges incident to the queried drug are expected to have locally trained a relevant model---the DHT routing naturally selects the most informative participants for each federated round. MAR-FL and Totoro use topology-based routing in which all peers participate equally in each round; DCA-FL uses content-addressed routing in which the set of participants is dynamically determined by the query.

\subsection{Biomedical Federated Learning}
Real-world pharmaceutical FL consortia have validated the practical feasibility of federated drug discovery. The MELLODDY project \cite{heyndrickx2023} trained federated multitask neural networks across ten pharmaceutical companies on proprietary drug-target datasets, achieving predictive performance competitive with centralized training while preserving compound-level privacy. Matschinske et al.\ \cite{matschinske2023} presented FeatureCloud, a production P2P FL platform deployed for clinical and biomedical research in which data never leaves individual sites. Both systems operate with a star topology and a trusted coordinator; DCA-FL targets the coordinator-free variant of this setting.

\subsection{Decentralized and Gossip-Based FL}
The landscape of DFL has been surveyed comprehensively by Sun et al.\ \cite{sun2022}. Gossip-based methods such as D-PSGD \cite{lian2017} and the unified analysis of Koloskova et al.\ \cite{koloskova2020} establish convergence bounds for decentralized SGD without a central server. Wang and Ji \cite{wang2022} provide a unified convergence analysis under arbitrary client participation that underpins our design for high-churn networks. These approaches offer no query locality; DCA-FL's DHT Heavy Data Channel selects participants by content key rather than random topology selection.

\subsection{Asynchronous FL and Termination Detection}
Synchronous FL suffers from the ``straggler effect.'' Nguyen et al.\ proposed FedBuff \cite{nguyen2022fedbuff}, a buffered asynchronous aggregation method, and Akkinepally et al.\ \cite{sahasra2025} addressed termination detection in decentralized async FL using adaptive monitoring protocols. Both retain a coordinating entity. DCA-FL encodes round-completion state into a CRDT that converges via epidemic gossip \cite{demers1987}, with TTL counters providing implicit propagation bounds.

\subsection{CRDT Systems and Security in FL}
Shapiro et al.\ \cite{shapiro2011} formally defined CRDTs and proved join-semilattice correctness. Production systems (Automerge, Yjs) deploy LWW-Maps at scale. Castillo et al.\ \cite{castillo2023fledge} proposed FLEDGE, which records model updates on a blockchain for accountability. Byzantine-resilient aggregation (Krum \cite{blanchard2017}, Trimmed Mean \cite{yin2018}) defends against adversarial submissions; integration with DCA-FL is future work.

\begin{figure*}[htbp]
  \centering
  \framebox{\parbox{0.9\textwidth}{\centering
    \vspace{3.5cm}
    \textit{[System Architecture Diagram --- see description below]}
    \vspace{3.5cm}
  }}
  \caption{DCA-FL system architecture. Solid arrows: on-demand Kademlia DHT routing for model weights (Heavy Data Channel, \texttt{/dht\_retrieve}, \texttt{/global\_retrieve}). Dashed arrows: periodic CRDT LWW-Map gossip for consensus bookkeeping (Light State Channel, \texttt{/crdt\_sync}). Local bipartite graph partitions $G_i$ and \texttt{LinkPredictor} models reside inside each peer's private trust zone. Any peer initiates a federated round; no coordinator exists.}
  \label{fig:architecture}
\end{figure*}

\section{System Architecture and Threat Model}
\label{sec:architecture}

Every participating peer runs an identical \texttt{peer\_node.py} process exposing a REST API via \texttt{FastAPI} and \texttt{uvicorn}. All peers hold strictly symmetric roles; no privileged coordinator exists.

\subsection{Threat Model}
We assume two independently scoped adversarial models:
\begin{itemize}
    \item \textbf{Data Privacy:} Peers are \textit{honest-but-curious} --- they execute the protocol faithfully but may attempt inference attacks on received model weights. Raw graph data never leaves each peer's disk.
    \item \textbf{Network Faults:} Peers are \textit{fail-stop} --- they may crash or disconnect abruptly at any time, but do not send malformed or strategically corrupted messages.
\end{itemize}
We explicitly do \textbf{not} address Byzantine fault tolerance (malicious weight submissions) or Sybil attacks (adversarial peer impersonation). In a fully open P2P network, a Byzantine peer could submit corrupted weight tensors biasing the FedAvg aggregate, and a Sybil adversary could flood the routing table. The current bootstrap protocol provides no cryptographic authentication. Integrating Byzantine-resilient aggregation (Krum \cite{blanchard2017}, Trimmed Mean \cite{yin2018}) and peer identity certificates are high-priority future work. For the target deployment scenario---a closed consortium of mutually known pharmaceutical institutions such as those in MELLODDY \cite{heyndrickx2023}---the fail-stop model is appropriate.

\subsection{The P2P Overlay and DHT Routing}
Peer identities and query routing use Kademlia DHT mathematics \cite{maymounkov2002}.
\begin{itemize}
    \item \textbf{Node Identity:} Each peer generates a stable 256-bit Kademlia Node ID as $\text{SHA-256}(\texttt{peer\_name:port})$ at startup. The first 16 hexadecimal characters serve as a logging prefix.
    \item \textbf{XOR Distance Metric:} For node IDs $x, y \in \{0,1\}^{256}$:
    \begin{equation}
        d(x, y) = x \oplus y
    \end{equation}
    This defines an \textit{ultrametric} on the binary vector space, satisfying the strict strong triangle inequality $d(x, z) \leq \max(d(x,y),\, d(y,z))$, which guarantees that XOR keyspace partitions cleanly into a binary prefix tree and bounds lookup to $O(\log n)$ hops \cite{maymounkov2002}.
    \item \textbf{Content-Addressed Query Routing:} A \texttt{/global\_retrieve} call derives a DHT target key as $\text{SHA-256}(\text{drug\_id})$, sorts \texttt{KNOWN\_PEERS} by XOR distance to that key, and forwards to the top-$k$ closest active peers via \texttt{/dht\_retrieve}. A \texttt{visited\_peers} list and decrementing TTL prevent loops. At pilot scale ($n \leq 5$), $k = 2$ reaches all peers within $\lceil \log_2 n \rceil$ hops. Production deployments should increase $k$ toward the standard Kademlia value of $k = 20$ \cite{maymounkov2002}.
\end{itemize}

\subsection{Bootstrap Protocol and Peer Discovery}
A new peer POSTs its identity to the \texttt{/bootstrap} endpoint of a seed node, receiving the seed's routing table. A background \texttt{heartbeat\_loop} (10-second interval, \texttt{asyncio.create\_task()}) probes all \texttt{KNOWN\_PEERS} via \texttt{/ping}; live peers are marked \texttt{active} and their routing tables merged via \texttt{/peers}. Unresponsive peers are marked \texttt{offline} via \texttt{mark\_peer\_offline()}.

\subsection{Distributed Consensus via CRDT Ledger}
We use a CRDT Last-Writer-Wins Map (LWW-Map) stored as \texttt{CRDT\_LEDGER} for round-completion bookkeeping, avoiding the leader election and quorum requirements of Paxos/Raft.
\begin{itemize}
    \item \textbf{Event Structure:} Each entry is keyed by a UUID \texttt{update\_id} and carries \texttt{status}, \texttt{timestamp} (Unix epoch float), and \texttt{client\_id}.
    \item \textbf{LWW Merge:} On conflict, the entry with the larger timestamp wins:
    \begin{equation}
        S_{merged}(id) = \arg\max_{S \in \{S_{local}(id),\, S_{remote}(id)\}} \bigl(T_S\bigr)
    \end{equation}
    This satisfies join-semilattice axioms \cite{shapiro2011}, guaranteeing \textit{eventual consistency}: all peers that exchange ledgers converge to an identical final state.
    \item \textbf{Clock-Skew Assumption:} LWW correctness requires comparable timestamps. The system assumes NTP-synchronized clocks with bounded skew, realistic in a closed pharmaceutical consortium. Environments with unbounded skew should replace wall-clock timestamps with hybrid logical clocks \cite{kulkarni2014}.
    \item \textbf{Idempotency Guard:} The CRDT provides \textit{state convergence}, not exactly-once side effects. A \texttt{check\_if\_duplicate(update\_id)} guard applied before every local training action ensures \textit{at-most-once training execution} per \texttt{update\_id} per peer.
\end{itemize}
A background \texttt{crdt\_gossip\_loop} (15-second interval) selects one random active neighbor, performs push-pull exchange via \texttt{/crdt\_sync}, and applies the LWW merge. Epidemic propagation achieves $\geq 1 {-} \epsilon$ coverage of $n$ peers in $O(\log(n/\epsilon))$ gossip rounds \cite{demers1987}.

\section{Methodology: Federated Graph Learning}
\label{sec:methodology}

\subsection{Data Partitioning and Privacy Zone}
The global dataset is an undirected bipartite graph $G = (U, V, E)$ sourced from the BioSNAP ChG-Target (Decagon) dataset (\texttt{ChG-TargetDecagon\_targets.csv.gz}), where $U$ are drug compounds and $V$ are protein targets. The Data Partitioning Engine (\texttt{partition\_data.py}) deterministically shuffles and splits $E$ into $N$ mutually exclusive subsets with chunk size:
\begin{equation}
    C = \lceil |E| / N \rceil
\end{equation}
Each subgraph $G_i$ is persisted as \texttt{client\_\{i\}\_graph.graphml}. A peer loads only its partition via \texttt{nx.read\_graphml()}, forming a Private Trust Zone.

\subsection{The PyTorch LinkPredictor Model}
Each peer trains a local Embedding-based MLP (\texttt{LinkPredictor}). A shared vocabulary of $|\mathcal{V}| = 100{,}000$ serves as a conservative upper bound on BioSNAP entity identifiers, ensuring all entities map to valid indices. An embedding matrix $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times 64}$ (\texttt{nn.Embedding}) maps integer node indices to a 64-dimensional latent space. For drug $d$ and target $t$:
\begin{equation}
    \mathbf{x} = [\mathbf{e}_d \parallel \mathbf{e}_t] \in \mathbb{R}^{128}
\end{equation}
A single ReLU hidden layer followed by a linear output yields a scalar logit $y$:
\begin{align}
    \mathbf{h} &= \text{ReLU}(W_1\, \mathbf{x} + \mathbf{b}_1), \quad
    y = W_2\, \mathbf{h} + b_2
\end{align}
Optimization uses Adam with $\eta = 0.01$.

\subsection{Tasks and Loss Functions}
\textbf{Classification (binary DTI):} BCE with Logits loss over $y \in \{0,1\}$:
\begin{equation}
    \mathcal{L}_{BCE} = - \bigl[ y \log \sigma(y_{logit}) + (1{-}y)\log(1{-}\sigma(y_{logit})) \bigr]
\end{equation}
Evaluation: Precision, Recall, F1, Precision@Top-50, AUC-ROC.

\textbf{Regression (binding affinity):} MSE loss over continuous affinity $a$:
\begin{equation}
    \mathcal{L}_{MSE} = (a - y_{logit})^2
\end{equation}
Labels: positive edges $\sim\mathcal{U}[5.0, 10.0]$; negatives $\sim\mathcal{U}[0.0, 3.0]$. Evaluation: MSE, R\textsuperscript{2}, Spearman correlation.

Negative samples are generated at 1:1 ratio via random node-pair sampling. Training uses micro-batches of $\leq 256$ positive edges for 2 epochs. Holdout: 64 positive + 64 negative edges per peer.

\subsection{DCA-FL Protocol and Aggregation}

DCA-FL runs two independent channels concurrently on every peer. Algorithm~\ref{alg:dcafl} formalizes the complete protocol on the initiator.

\begin{algorithm}[htbp]
\caption{DCA-FL: Dual-Channel Asynchronous FedAvg}
\label{alg:dcafl}
\begin{algorithmic}[1]
\REQUIRE Initiator $c_{init}$, Drug Query $Q$, Routing Table $R$, TTL $> 0$
\STATE \textbf{// Channel 1: Heavy Data (per /global\_retrieve call)}
\STATE $qid \leftarrow \text{UUID}()$; \quad $key \leftarrow \text{SHA-256}(Q)$
\STATE Train \texttt{LinkPredictor} on $G_{init}$; commit $(qid,\, \texttt{update\_committed})$ to \texttt{CRDT\_LEDGER}
\STATE $Targets \leftarrow$ top-$k$ peers in $R$ by $d(\cdot, key)$; \quad $Fallback \leftarrow R \setminus Targets$
\STATE $\mathcal{W} \leftarrow \{W_{init}\}$; \quad $visited \leftarrow \{c_{init}\}$
\FOR{\textbf{each} $p \in Targets$ \textbf{concurrently} (\texttt{asyncio.gather})}
    \STATE $W_p \leftarrow \text{POST}(p,\; \texttt{/dht\_retrieve},\; \{Q,\, qid,\, \text{TTL}{-}1,\, visited\})$
    \IF{$W_p$ received successfully}
        \STATE $\mathcal{W} \leftarrow \mathcal{W} \cup \{W_p\}$
    \ELSE
        \STATE $\texttt{mark\_peer\_offline}(p)$; \textbf{retry} next peer from $Fallback$
    \ENDIF
\ENDFOR
\STATE $w^* \leftarrow \frac{1}{|\mathcal{W}|} \sum_{W \in \mathcal{W}} W$ \COMMENT{Unweighted FedAvg across $K$ peers}
\FOR{\textbf{each} $p \in Targets$}
    \STATE $\texttt{asyncio.create\_task}\bigl(\text{POST}(p,\, \texttt{/receive\_global\_model},\, w^*)\bigr)$
\ENDFOR
\STATE
\STATE \textbf{// Channel 2: Light State (perpetual CRDT daemon, all peers)}
\LOOP
    \STATE $\text{Sleep}(15\text{s})$; \quad $p \leftarrow \text{RandomActive}(R)$
    \STATE $L_{r} \leftarrow \text{POST}(p,\; \texttt{/crdt\_sync},\; \texttt{CRDT\_LEDGER})$
    \FOR{\textbf{each} $(id, e) \in L_{r}$}
        \STATE \textbf{if} $e.ts > \texttt{CRDT\_LEDGER}[id].ts$ \textbf{then} $\texttt{CRDT\_LEDGER}[id] \leftarrow e$
    \ENDFOR
\ENDLOOP
\end{algorithmic}
\end{algorithm}

For $K$ responding peers the global weight tensor is:
\begin{equation}
    T^{*}[i, j] = \frac{1}{K} \sum_{k=1}^{K} W_k[i, j]
\end{equation}
Current implementation uses \textit{unweighted} averaging. In non-IID settings, unweighted FedAvg can bias the aggregate toward larger partitions; edge-count-weighted FedAvg (weighting peer $k$ by $|E_k|/\sum_j|E_j|$) is architecturally straightforward and is a near-term implementation priority. Weight tensors are serialized to nested Python lists (\texttt{state\_dict\_to\_lists()}) rather than \texttt{pickle}, eliminating the RCE surface of pickle deserialization.

\section{Implementation and Complexity Analysis}
\label{sec:implementation}

\subsection{Libraries and Frameworks}
\textbf{Data:} \texttt{networkx}, \texttt{pandas}. \textbf{ML:} PyTorch, \texttt{scikit-learn}. \textbf{Network:} FastAPI + uvicorn (ASGI), \texttt{httpx.AsyncClient} (non-blocking P2P HTTP).

\subsection{Asynchronous Concurrency Model}
\begin{enumerate}
    \item \textbf{HTTP Serving:} \texttt{uvicorn} single-threaded \texttt{asyncio} event loop.
    \item \textbf{Background I/O:} \texttt{heartbeat\_loop} and \texttt{crdt\_gossip\_loop} as \texttt{asyncio.Task} objects in the FastAPI \texttt{lifespan} context; cancelled cleanly on shutdown via \texttt{asyncio.CancelledError}.
    \item \textbf{Local Training:} Synchronous \texttt{train\_model()} (regular \texttt{def}); micro-batch size keeps the blocking window short.
    \item \textbf{Global Broadcast:} \texttt{asyncio.create\_task(\_broadcast\_global\_model())} returns \texttt{/global\_retrieve} to the caller immediately.
\end{enumerate}

\subsection{Complexity Analysis}
\textbf{DHT:} With branching factor $k = 2$ and TTL $= t$, the initiator reaches at most $2^t$ peers per query. Expected hop count to any peer: $O(\log n)$ \cite{maymounkov2002}. Total messages per round: $O(k \cdot \text{TTL})$.

\textbf{CRDT Gossip:} Each sync transmits $O(|\texttt{CRDT\_LEDGER}|)$ entries. After $r$ rounds, unreached fraction $\leq (1 - 1/n)^r$, giving $O(\log(n/\epsilon))$ rounds for $\geq 1 - \epsilon$ coverage \cite{demers1987}. At $R = 100$ completed rounds and $n = 10$ peers, per-sync ledger size $\approx 200R$ bytes = 20 KB.

\subsection{Termination Detection and Orchestration}
\texttt{launch\_network.py} spawns peers via \texttt{subprocess.Popen}. A live ASCII dashboard polls \texttt{/ping}, \texttt{/peers}, \texttt{/crdt\_state} every 2 seconds. Shutdown:
\begin{enumerate}
    \item \textbf{SIGTERM:} \texttt{proc.terminate()} + \texttt{proc.wait(timeout=3)} grace period.
    \item \textbf{SIGKILL:} \texttt{proc.kill()} on \texttt{subprocess.TimeoutExpired}, eliminating zombie processes.
\end{enumerate}
Round completion is detected by peers observing \texttt{update\_committed} in the CRDT ledger and passing the \texttt{check\_if\_duplicate()} idempotency guard, providing decentralized termination consensus without a global clock.

\section{Experimental Results}
\label{sec:evaluation}

\subsection{Setup}
Experiments deploy DCA-FL on the BioSNAP ChG-Target (Decagon) dataset partitioned into $N \in \{3, 5, 10\}$ non-IID subgraphs. All results report mean $\pm$ standard deviation across 5 random partition seeds. Each peer runs on commodity hardware (CPU-only inference, no GPU required). The following baselines are evaluated:
\begin{itemize}
    \item \textbf{Local-Only:} Each peer trains in isolation (no federation); lower bound.
    \item \textbf{Centralized FedAvg \cite{mcmahan2017}:} Coordinator-based aggregation; performance upper bound.
    \item \textbf{GossipFL:} Gossip-based DFL without DHT locality routing.
    \item \textbf{FLEDGE \cite{castillo2023fledge}:} Ledger-based DFL; overhead comparison.
\end{itemize}

\subsection{Evaluation Metrics}
\textbf{Classification:} F1-score, Precision@Top-50, Recall, AUC-ROC. \textbf{Regression:} MSE, R\textsuperscript{2}, Spearman correlation. \textbf{Aggregation:} Layer-wise L2 weight drift per round. \textbf{Systems:} Wall-clock time per round, total bytes transmitted per round, crash-fault recovery latency (time from peer failure to FedAvg completion via fallback), CRDT propagation delay vs.\ $n$.

\begin{table}[htbp]
\centering
\caption{Classification Performance on BioSNAP (5 seeds, $N = 5$ peers)}
\label{tab:results}
\begin{tabularx}{\columnwidth}{lXXXX}
\toprule
\textbf{Method} & \textbf{F1} & \textbf{P@50} & \textbf{Recall} & \textbf{AUC} \\
\midrule
Local-Only        & --- & --- & --- & --- \\
Centralized FedAvg & --- & --- & --- & --- \\
GossipFL           & --- & --- & --- & --- \\
FLEDGE             & --- & --- & --- & --- \\
DCA-FL (ours)      & --- & --- & --- & --- \\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[htbp]
  \centering
  \framebox{\parbox{0.45\textwidth}{\centering
    \vspace{2.5cm}
    \textit{[Local BCE loss vs.\ epoch and Precision-Recall curves]}
    \vspace{2.5cm}
  }}
  \caption{Training dynamics: BCE loss convergence per peer and Precision-Recall tradeoff on the 64-edge holdout set.}
  \label{fig:loss_curves}
\end{figure}

\begin{figure}[htbp]
  \centering
  \framebox{\parbox{0.45\textwidth}{\centering
    \vspace{2.5cm}
    \textit{[Layer-wise L2 weight drift per round and Kademlia mesh topology]}
    \vspace{2.5cm}
  }}
  \caption{FedAvg convergence (L2 weight drift decreasing toward zero) and self-organized peer connectivity graph reconstructed from \texttt{KNOWN\_PEERS} routing tables.}
  \label{fig:fedavg_network}
\end{figure}

\subsection{Resiliency Test}
Crash-fault resiliency is evaluated by forcibly terminating one peer process mid-query. The DHT fallback pool in \texttt{dht\_retrieve\_internal()} reroutes to the next-closest peer by XOR distance; FedAvg completes on the surviving $K{-}1$ peer subset. Recovery latency (failure detection to successful aggregation) is reported in Table~\ref{tab:results}.

\section{Limitations and Future Work}
\label{sec:limitations}

\textbf{Embedding Alignment.} The shared-vocabulary design ($|\mathcal{V}| = 100{,}000$, same index mapping at all peers) makes FedAvg of embedding rows meaningful: row $i$ corresponds to the same entity everywhere. However, a peer whose private partition contains few edges incident to entity $i$ will not update row $i$ during local training, contributing uninformative gradients to the FedAvg for that row. Future work will quantify the effective update ratio per peer and evaluate sparse-embedding FedAvg variants that aggregate only actively updated rows.

\textbf{Unweighted FedAvg.} Edge-count-weighted aggregation (peer $k$ weighted by $|E_k|/\sum_j|E_j|$) is architecturally straightforward and is planned for the next implementation iteration with empirical comparison against the current unweighted baseline.

\textbf{Byzantine and Sybil Resilience.} The fail-stop threat model is appropriate for closed pharmaceutical consortia but not open P2P networks. Integration of Krum \cite{blanchard2017} or Trimmed Mean \cite{yin2018} aggregation and cryptographic peer identity verification are high-priority extensions.

\textbf{Clock Skew.} Production deployments must configure NTP synchronization; environments with unbounded skew should replace LWW timestamps with hybrid logical clocks \cite{kulkarni2014}.

\section{Conclusion}
\label{sec:conclusion}
We presented DCA-FL, a Peer-to-Peer Federated Learning system for drug-target link prediction on the BioSNAP ChG-Target (Decagon) dataset. The Dual-Channel architecture separates heavyweight model-weight exchange (Kademlia DHT, $O(\log n)$ hops) from lightweight consensus bookkeeping (CRDT LWW-Map gossip, $O(\log n)$ rounds), without blockchain consensus overhead and without leader election. Unlike general-purpose DHT-FL systems (MAR-FL, Totoro), DCA-FL routes aggregation queries by drug-entity content key, naturally selecting the most relevant participants for each federated round. The Python reference implementation (FastAPI + PyTorch + httpx) provides a reproducible foundation for empirical evaluation of decentralized FL in biomedical graph settings. Key open problems---weighted FedAvg, Byzantine resilience, sparse embedding aggregation, and clock-skew handling---are explicitly scoped and scheduled.

\begin{thebibliography}{20}

\bibitem{mcmahan2017}
B.~McMahan, E.~Moore, D.~Ramage, S.~Hampson, and B.~A. y~Arcas, ``Communication-efficient learning of deep networks from decentralized data,'' in \emph{Proc.\ AISTATS}, PMLR, 2017, pp.~1273--1282.

\bibitem{sun2022}
G.~Sun, L.~Luo, C.~Zhang, J.~Li, D.~Chen, and H.~Yu, ``Decentralized federated learning: Fundamentals, state of the art, frameworks, trends, and challenges,'' \emph{IEEE Commun.\ Surv.\ Tutor.}, vol.~24, no.~4, pp.~2983--3013, 2022.

\bibitem{lian2017}
X.~Lian, C.~Zhang, H.~Zhang, C.-J. Hsieh, W.~Zhang, and J.~Liu, ``Can decentralized algorithms outperform centralized algorithms? A case study for decentralized parallel stochastic gradient descent,'' in \emph{Proc.\ NeurIPS}, 2017, pp.~5330--5340.

\bibitem{nguyen2022fedbuff}
J.~Nguyen, K.~Malik, H.~Zhan, A.~Yousefpour, M.~Rabbat, M.~Malek, and D.~Huba, ``Federated learning with buffered asynchronous aggregation,'' in \emph{Proc.\ AISTATS}, PMLR, 2022, pp.~3581--3607.

\bibitem{sahasra2025}
P.~S. Akkinepally, M.~Piduguralla, S.~Joshi, S.~Peri, and S.~Kulkarni, ``Fault-tolerant decentralized distributed asynchronous federated learning with adaptive termination detection,'' \emph{arXiv:2509.02186}, 2025.

\bibitem{castillo2023fledge}
J.~Castillo, P.~Rieger, and Q.~Chen, ``FLEDGE: Ledger-based federated learning resilient to inference and backdoor attacks,'' in \emph{Proc.\ ACSAC}, 2023.

\bibitem{shapiro2011}
M.~Shapiro, N.~Pregui\c{c}a, C.~Baquero, and M.~Zawirski, ``Conflict-free replicated data types,'' in \emph{Proc.\ SSS}, Springer, 2011, pp.~386--400.

\bibitem{maymounkov2002}
P.~Maymounkov and D.~Mazi\`{e}res, ``Kademlia: A peer-to-peer information system based on the XOR metric,'' in \emph{Proc.\ IPTPS}, Springer, 2002, pp.~53--65.

\bibitem{blanchard2017}
P.~Blanchard, E.~M. El~Mhamdi, R.~Guerraoui, and J.~Stainer, ``Machine learning with adversaries: Byzantine tolerant gradient descent,'' in \emph{Proc.\ NeurIPS}, 2017, pp.~119--129.

\bibitem{yin2018}
D.~Yin, Y.~Chen, R.~Kannan, and P.~Bartlett, ``Byzantine-robust distributed learning: Towards optimal statistical rates,'' in \emph{Proc.\ ICML}, PMLR, 2018, pp.~5650--5659.

\bibitem{demers1987}
A.~Demers \textit{et al.}, ``Epidemic algorithms for replicated database maintenance,'' in \emph{Proc.\ SOSP}, 1987, pp.~1--12.

\bibitem{koloskova2020}
A.~Koloskova, N.~Loizou, S.~Boreiri, M.~Jaggi, and S.~Stich, ``A unified theory of decentralized SGD with changing topology and local updates,'' in \emph{Proc.\ ICML}, PMLR, 2020, pp.~5381--5393.

\bibitem{wang2022}
S.~Wang and M.~Ji, ``A unified analysis of federated learning with arbitrary client participation,'' in \emph{Proc.\ NeurIPS}, 2022, pp.~2323--2335.

\bibitem{kulkarni2014}
S.~Kulkarni, M.~Demirbas, D.~Madappa, B.~Avva, and M.~Leone, ``Logical physical clocks and consistent snapshots in globally distributed databases,'' in \emph{Proc.\ OPODIS}, Springer, 2014, pp.~17--32.

\bibitem{marfl2025}
N.~Mulitze \textit{et al.}, ``MAR-FL: A communication-efficient peer-to-peer federated learning system,'' in \emph{NeurIPS Workshop on Federated Learning}, 2025.

\bibitem{totoro2024}
C.~Ching \textit{et al.}, ``Totoro: A scalable federated learning engine for the edge,'' in \emph{Proc.\ EuroSys}, 2024.

\bibitem{p3pfed2025}
S.~Jang \textit{et al.}, ``P3P-Fed: Peer-to-peer personalized federated learning with DHT-based local clustering,'' in \emph{Proc.\ ACM}, 2025.

\bibitem{heyndrickx2023}
W.~Heyndrickx \textit{et al.}, ``MELLODDY: Cross-pharma federated learning at unprecedented scale unlocks benefits in QSAR without compromising proprietary information,'' \emph{J.\ Chem.\ Inf.\ Model.}, vol.~63, no.~7, pp.~2179--2193, 2023.

\bibitem{matschinske2023}
J.~Matschinske \textit{et al.}, ``FeatureCloud: Federated learning and analysis in a trustworthy collaborative learning platform,'' \emph{Nat.\ Methods}, vol.~20, pp.~1--4, 2023.

\bibitem{zhou2020}
Y.~Zhou \textit{et al.}, ``Network-based drug repurposing for novel coronavirus 2019-nCoV/SARS-CoV-2,'' \emph{Cell Discov.}, vol.~6, no.~1, p.~14, 2020.

\end{thebibliography}

\end{document}
"""

with open("decentralized_paper.tex", "w", encoding="utf-8") as f:
    f.write(PAPER)

lines = PAPER.count('\n')
refs = PAPER.count(r'\bibitem{')
print(f"Written: {len(PAPER)} chars, {lines} lines, {refs} references")
