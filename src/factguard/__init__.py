"""factguard: catch an LLM inventing or twisting numbers when it writes from data.

    from factguard import check
    report = check(llm_text, source_json)
    report.ok, report.score, report.unsupported
"""
from .core import Checker, Report, Verdict, assert_grounded, check, flatten
from .extract import Claim, extract_claims

__all__ = ["check", "assert_grounded", "Checker", "Report", "Verdict", "flatten",
           "extract_claims", "Claim"]
__version__ = "0.2.1"
