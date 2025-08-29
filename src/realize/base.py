from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class RealizeConfig:
    style: Optional[Dict[str, Any]] = None      # politeness, formality, etc.
    constraints: Optional[Dict[str, Any]] = None # grammar masks, lex constraints
    timeout_ms: int = 4000
    profile: str = "standard"                    # or "neural" | "hybrid"

class Realizer:
    def realize(self, src_eil, tgt_lang: str, *, binder=None, config: RealizeConfig = RealizeConfig()):
        raise NotImplementedError
