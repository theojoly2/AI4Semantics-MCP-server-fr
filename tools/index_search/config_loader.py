"""
Configuration loader - chargé UNE SEULE FOIS au démarrage du serveur.
Évite de recharger vocabularies.xlsx et de régénérer le prompt à chaque appel.
"""

from pathlib import Path
import pandas as pd
from typing import Optional

# ============================================================================
# CACHE GLOBAL
# ============================================================================

_CACHE: dict = {}

# ============================================================================
# VOCABULARIES - Chargé au démarrage du module
# ============================================================================

def _get_vocab_file() -> Path:
    """Retourne le chemin vers vocabularies.xlsx"""
    return Path(__file__).resolve().parent / "vocabularies.xlsx"


def _load_vocabularies_from_excel() -> list[str]:
    """
    Fonction privée appelée UNE SEULE FOIS au démarrage du module.
    Retourne la liste des noms de vocabulaires.
    """
    vocab_file = _get_vocab_file()
    
    if not vocab_file.exists():
        print(f"⚠️  {vocab_file} introuvable, vocabulaires par défaut")
        return ["CAV", "CBV", "CCCEV", "CLV", "CPEV", "CPOV", "CPSV", "CPV"]
    
    try:
        df = pd.read_excel(vocab_file)
        
        if "NAME" not in df.columns:
            raise ValueError("Colonne 'NAME' absente")
        
        vocab_names = [
            str(name).strip() 
            for name in df["NAME"].tolist() 
            if pd.notna(name) and str(name).strip() and str(name).strip().lower() != "nan"
        ]
        
        if not vocab_names:
            raise ValueError("Aucun vocabulaire trouvé")
        
        print(f"✅ Chargé {len(vocab_names)} vocabulaires depuis {vocab_file.name}")
        return vocab_names
        
    except Exception as e:
        print(f"⚠️  Erreur chargement vocabulaires: {e}")
        return ["CAV", "CBV", "CCCEV", "CLV", "CPEV", "CPOV", "CPSV", "CPV"]


def _load_vocabulary_descriptions() -> str:
    """
    Fonction privée pour générer la section de guidance du prompt.
    Appelée UNE SEULE FOIS au démarrage du module.
    """
    vocab_file = _get_vocab_file()
    
    if not vocab_file.exists():
        return "No vocabulary metadata available."
    
    try:
        df = pd.read_excel(vocab_file)
        
        if not {"NAME", "DESCRIPTION"}.issubset(df.columns):
            return "Vocabulary metadata incomplete (missing NAME or DESCRIPTION)."
        
        lines = []
        lines.append("AVAILABLE VOCABULARIES (with descriptions):\n")
        
        for _, row in df.iterrows():
            name = str(row["NAME"]).strip()
            description = str(row["DESCRIPTION"]).strip()
            
            # Ignorer les lignes vides ou NaN
            if (not name or name.lower() == "nan" or 
                not description or description.lower() == "nan"):
                continue
            
            # Tronquer les descriptions très longues
            if len(description) > 300:
                description = description[:297] + "..."
            
            lines.append(f"   • {name}: {description}")
        
        lines.append("\nINFERENCE RULES:")
        lines.append("   - Read the user's query carefully and identify domain keywords")
        lines.append("   - Match those keywords against the vocabulary descriptions above")
        lines.append("   - Select 1-3 most relevant vocabularies based on semantic match")
        lines.append("   - If multiple match equally, prefer Core vocabularies (CAV, CBV, CCCEV, CLV, CPEV, CPOV, CPSV, CPV)")
        lines.append("   - If exploratory or domain-ambiguous, omit vocabularies to search broadly")
        lines.append("   - Trust the built-in fallback: retrieve_documents will retry without filters if needed")
        
        return '\n'.join(lines)
    
    except Exception as e:
        print(f"⚠️  Erreur génération prompt vocabularies: {e}")
        return "No vocabulary metadata available."


def warmup_config() -> dict:
    """
    Précharge la configuration si ce n'est pas déjà fait.
    Retourne le cache global.
    """
    global _CACHE
    
    if _CACHE:
        return _CACHE
    
    print("🔄 Chargement de la configuration...")
    
    # Charger les vocabulaires
    _CACHE["VOCABULARIES"] = _load_vocabularies_from_excel()
    
    # Générer la section de guidance pour le prompt
    _CACHE["VOCABULARY_GUIDANCE"] = _load_vocabulary_descriptions()
    
    print("✅ Configuration chargée et mise en cache")
    
    return _CACHE


def get_vocabularies() -> list[str]:
    """Retourne la liste des vocabulaires disponibles"""
    return warmup_config()["VOCABULARIES"]


def get_vocabulary_guidance() -> str:
    """Retourne la guidance des vocabulaires pour le prompt"""
    return warmup_config()["VOCABULARY_GUIDANCE"]


# ============================================================================
# CHARGEMENT AU DÉMARRAGE DU MODULE (exécuté UNE SEULE FOIS à l'import)
# ============================================================================

# Charger au premier import du module
VOCABULARIES = get_vocabularies()
VOCABULARY_GUIDANCE = get_vocabulary_guidance()

# ============================================================================
# EXPORTS PUBLICS
# ============================================================================

__all__ = [
    "VOCABULARIES", 
    "VOCABULARY_GUIDANCE",
    "get_vocabularies",
    "get_vocabulary_guidance",
    "warmup_config",
]
