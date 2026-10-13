"""The learned editor: a prior over moves (not over solutions), trained only on the search's own history.

  featurize  circuit -> arrays; Move <-> factorized action indices
  model      JAX encoder (message passing + relation-biased attention) and factorized move heads; dream head
  proposal   LearnedProposal (a moves.Proposal) and the sampler
  metrics    collapse instruments (entropy, top-k mass, KL to uniform)
  store      CircuitStore (circuits by hash, saved beside an edit log) and a recording archive
  train      hindsight dataset from edit logs, training, held-out evaluation, checkpoints
  dream      random circuits simulated on the stand-in substrate; inverse-model pretraining; generator stub
"""
from .model import EditorConfig, init_params, load_params, save_params  # noqa: F401
from .proposal import LearnedProposal  # noqa: F401
from .store import CircuitStore, RecordingArchive  # noqa: F401
