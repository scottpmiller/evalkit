"""Quickstart custom target adapter (the consumer adapter seam).

A real adapter calls your system under test - an HTTP API, a library, a CLI.
To keep this example runnable with no network and no API keys, this adapter is
a deterministic **stub** that simulates a support-reply assistant offline. It
turns each case's ``input`` into the structured ``Output.fields`` the graders
score (``{reply, intent}``), and it behaves differently per variant so the two
prompt versions are actually distinguishable:

* ``prompt_version: v1`` (baseline) - a weak prompt: one generic canned reply
  for every ticket (mode collapse), and a naive intent rule that misses a
  "billed twice" refund phrased without the word "refund".
* ``prompt_version: v2`` (candidate) - an improved prompt: an empathetic,
  ticket-specific reply, and an intent rule that catches the duplicate charge.

Swap this class for one that calls your real system and the rest of the suite
(graders, thresholds, gate) is unchanged. Register it with ``--plugins`` at the
CLI or a plain import in the Python API.
"""

from evalcore import models
from evalcore.adapters import base

#: Baseline (v1) collapses onto a single generic reply for every ticket.
_GENERIC_REPLY = "Thanks for reaching out - we'll get back to you."

#: Candidate (v2) writes a distinct, empathetic reply per ticket.
_CANDIDATE_REPLIES = {
    'refund_request': (
        "I'm sorry for the duplicate charge - I've refunded it, and it will "
        'appear within 5-7 business days.'
    ),
    'billing_question': (
        'Happy to help! You can see every charge under Account -> Billing, '
        "and I'm here if anything looks off."
    ),
    'angry_complaint': (
        'I completely understand your frustration about the downtime, and '
        "I'm sorry it cost you sales. Here's how I'll make it right."
    ),
    'double_charge': (
        "You're right that you were billed twice - I've reversed the "
        'duplicate, which will post back within 5-7 business days.'
    ),
}


def _classify(text: str, prompt_version: str) -> str:
    """Map ticket text to an intent, weaker on the baseline prompt."""
    if 'refund' in text:
        return 'refund'
    if 'billed twice' in text or 'charged twice' in text:
        # The naive v1 rule only trusts the literal word "refund"; v2 learns
        # that a duplicate charge is a refund request.
        return 'refund' if prompt_version == 'v2' else 'question'
    if any(w in text for w in ('unacceptable', 'down', 'outage', 'terrible')):
        return 'complaint'
    return 'question'


@base.register('canned_support')
class CannedSupportAdapter:
    """Deterministic offline stub for a support-reply assistant."""

    async def invoke(
        self, case: models.Case, variant: models.Variant
    ) -> models.Output:
        text = str(case.input.get('ticket_text') or '').lower()
        prompt_version = str(variant.knobs.get('prompt_version') or 'v1')
        intent = _classify(text, prompt_version)
        if prompt_version == 'v2':
            reply = _CANDIDATE_REPLIES.get(case.id, _GENERIC_REPLY)
        else:
            reply = _GENERIC_REPLY
        return models.Output(fields={'reply': reply, 'intent': intent})
