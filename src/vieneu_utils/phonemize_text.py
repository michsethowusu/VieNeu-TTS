"""
Phonemization module for VieNeu-TTS.
Delegates Vietnamese G2P to the sea-g2p library.
For other languages (e.g. Twi), falls back to espeak-ng via the phonemizer library.
"""

import functools
import logging
from typing import Optional
from sea_g2p import SEAPipeline, G2P, Normalizer

logger = logging.getLogger("Vieneu.Phonemizer")

# ---------------------------------------------------------------------------
# Espeak backend for non-Vietnamese languages
# ---------------------------------------------------------------------------

# Friendly language name → espeak language code
_ESPEAK_LANG_MAP = {
    "twi":  "lfn",
    "tw":   "lfn",
    "akan": "lfn",
}

def _to_espeak_code(lang: str) -> str:
    return _ESPEAK_LANG_MAP.get(lang.lower(), lang)

def _espeak_phonemize(text: str, lang: str) -> str:
    from phonemizer import phonemize as _ph
    return _ph(
        text,
        backend="espeak",
        language=_to_espeak_code(lang),
        with_stress=True,
        preserve_punctuation=True,
    )

def _espeak_phonemize_batch(texts: list, lang: str) -> list:
    from phonemizer import phonemize as _ph
    result = _ph(
        texts,
        backend="espeak",
        language=_to_espeak_code(lang),
        with_stress=True,
        preserve_punctuation=True,
        njobs=1,
    )
    return result if isinstance(result, list) else [result]

# ---------------------------------------------------------------------------
# Shared singletons (instantiation is lazy-safe and thread-safe via GIL)
# ---------------------------------------------------------------------------
_pipeline: SEAPipeline = None
_g2p: G2P = None
_normalizer: Normalizer = None

def _get_pipeline() -> SEAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SEAPipeline(lang="vi")
    return _pipeline

def _get_g2p() -> G2P:
    global _g2p
    if _g2p is None:
        _g2p = G2P(lang="vi")
    return _g2p

def _get_normalizer() -> Normalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = Normalizer()
    return _normalizer

# ---------------------------------------------------------------------------
# Public API  (same signatures as before — callers don't need to change)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _phonemize_cached(text: str) -> str:
    """Cached single-text phonemization (normalize + G2P)."""
    return _get_pipeline().run(text)


def phonemize_text(text: str, lang: str = "vi") -> str:
    """Normalize and phonemize a single text string."""
    if lang != "vi":
        return _espeak_phonemize(text, lang)
    return _phonemize_cached(text)


def phonemize_batch(
    texts: list,
    skip_normalize: bool = False,
    phoneme_dict: dict = None,
    lang: str = "vi",
    **kwargs,
) -> list:
    """
    Phonemize multiple texts.

    Args:
        texts:          List of input strings.
        skip_normalize: If True, skip normalization (Vietnamese only).
        phoneme_dict:   Optional custom {word: phoneme} override (Vietnamese only).
        lang:           Language code. Use 'vi' for Vietnamese (default),
                        'twi' for Twi/Akan, or any espeak language code.
    """
    if not texts:
        return []

    if lang != "vi":
        return _espeak_phonemize_batch(texts, lang)

    g2p = _get_g2p()
    if skip_normalize:
        return g2p.phonemize_batch(texts, phoneme_dict=phoneme_dict)
    else:
        normalizer = _get_normalizer()
        normalized = [normalizer.normalize(t) for t in texts]
        return g2p.phonemize_batch(normalized, phoneme_dict=phoneme_dict)


def phonemize_with_dict(
    text: str,
    phoneme_dict: dict = None,
    skip_normalize: bool = False,
    lang: str = "vi",
) -> str:
    """
    Phonemize a single text, optionally with a custom word→phoneme mapping.

    For non-Vietnamese languages, phoneme_dict and skip_normalize are ignored
    and espeak-ng is used directly.
    """
    if lang != "vi":
        return _espeak_phonemize(text, lang)
    if phoneme_dict is not None:
        return phonemize_batch(
            [text], skip_normalize=skip_normalize, phoneme_dict=phoneme_dict
        )[0]
    if skip_normalize:
        return _get_g2p().phonemize_batch([text])[0]
    return _phonemize_cached(text)


def phonemize_to_chunks(
    text: str,
    max_chars: int = 256,
    min_chunk_size: int = 10,
    source_max_chars: Optional[int] = None,
    skip_normalize: bool = False,
    phoneme_dict: dict = None,
):
    """
    Convert long raw text into bounded phoneme chunks.

    Some dependencies in the normalization/tokenization stack use Rust regex
    engines with backtracking limits. Split before those stages so DOCX-sized
    inputs are never passed to a single regex operation.
    """
    from vieneu_utils.core_utils import split_text_into_chunks, split_into_chunks_v2

    if not text:
        return []

    source_limit = source_max_chars or max_chars
    raw_chunks = split_text_into_chunks(text, max_chars=source_limit)
    if not raw_chunks:
        return []

    if skip_normalize:
        normalized_chunks = raw_chunks
    else:
        normalizer = _get_normalizer()
        normalized_chunks = [normalizer.normalize(chunk) for chunk in raw_chunks]

    phonemes = phonemize_batch(
        normalized_chunks,
        skip_normalize=True,
        phoneme_dict=phoneme_dict,
    )

    phone_chunks = []
    for chunk_phonemes in phonemes:
        phone_chunks.extend(
            split_into_chunks_v2(
                chunk_phonemes,
                max_chunk_size=max_chars,
                min_chunk_size=min_chunk_size,
            )
        )
    return phone_chunks


# ---------------------------------------------------------------------------
# CLI helper (python -m vieneu_utils.phonemize_text "some text")
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    test_text = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Giá SP500 hôm nay là 4.200,5 điểm."
    )
    print(f"Output: {phonemize_text(test_text)}")