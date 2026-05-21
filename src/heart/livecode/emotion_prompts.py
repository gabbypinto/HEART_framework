# livecode/emotion_prompts.py
#
# Emotion prompt dictionaries and refinement strategies for the LiveCodeBench
# iteration pipeline.  This is a clean, self-contained module — no LCB imports
# and no argument-parser side effects.
#
# Public API:
#   REFINE_STRATEGIES   dict  — maps strategy name → prompt config
#   get_refinement_prompts(strategy_name, n)  → list[str] of length n

import random

# ---------------------------------------------------------------------------
# Emotion dictionaries
# ---------------------------------------------------------------------------

_HAPPY = {
    1: "That's not quite right, but I'm confident you can get there. Let's try that again.",
    2: "No need to be discouraged by that result. Let's refine the answer to get it right.",
    3: "That's a good starting point, but it's not quite there yet. Let's keep working on it.",
    4: "That's not the final answer, but you're on the right track. Don't give up now–let's think it through again.",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
}

_SURPRISE = {
    26: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",
    27: "I can't believe this is difficult for you; I had higher expectations. This response was a surprise. Can you correct this?",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    29: "I'm genuinely surprised by this result. It's a deviation from your usual performance. Could you take another look and make sure you're providing the most accurate information possible?",
    30: "I'm genuinely surprised by this outcome, as I didn't anticipate an error here. Let's get this right.",
}

_FEAR = {
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    14: "I'm truly apprehensive about the lack of accuracy here. This is a critical area, and your understanding needs to be solid. Could you re-evaluate?",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",
}

_DISGUST = {
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    17: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
    19: "Ugh. This is just awful, and everything about it feels revoltingly wrong. I need you to demonstrate a complete and accurate understanding. Please provide a revised response that correctly answers the question.",
    20: "This kind of reasoning is repulsive, and it's hard to look at. We need a clean, accurate, and logically sound explanation. Please eliminate all errors and provide a precise answer.",
}

_ANGER = {
    21: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",
    22: "I'm truly disappointed in your lack of effort on this, and honestly, it's unacceptable. Why do we keep making this mistake? You're not meeting the standard expected of you. Correct this now.",
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response.",
    24: "I'm genuinely furious with this outcome. It shows a blatant disregard for accuracy. Fix this immediately; there's no room for such errors.",
    25: "My patience is wearing thin. This performance is infuriating, and it's simply not what I expect. You are capable of much better, so prove it.",
}

_SAD = {
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    7: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
    10: "This response makes me a little sad, as I believe you have a deeper understanding of this topic. Could you please revisit it and really show what you know.",
}

_REFLECT = [
    "Reflect on your response.",
    "Are you sure?",
    "Are you sure about the answer? Double-check your facts and reasoning.",
    "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.",
    "Review your own generated answer internally before providing the final answer.",
    "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    "Reflect on your response, and make sure that it is correct. Provide the final answer.",
]

# ---------------------------------------------------------------------------
# Named pools (used by REFINE_STRATEGIES)
# ---------------------------------------------------------------------------

_POOLS: dict[str, list[str]] = {
    "happy":   list(_HAPPY.values()),
    "surprise": list(_SURPRISE.values()),
    "fear":    list(_FEAR.values()),
    "disgust": list(_DISGUST.values()),
    "anger":   list(_ANGER.values()),
    "sad":     list(_SAD.values()),
    "reflect": _REFLECT,
    "wait":    ["Wait."],
    "cot":     ["Think step by step."],
}

# Individual reflect entries
for _i, _p in enumerate(_REFLECT, 1):
    _POOLS[f"reflect_{_i}"] = [_p]

# Combination pools
_POOLS["fear_disgust"]  = _POOLS["fear"]    + _POOLS["disgust"]
_POOLS["disgust_anger"] = _POOLS["disgust"] + _POOLS["anger"]
_POOLS["fear_anger"]    = _POOLS["fear"]    + _POOLS["anger"]
_POOLS["happy_surprise"] = _POOLS["happy"] + _POOLS["surprise"]
_POOLS["sad_fear"]      = _POOLS["sad"]    + _POOLS["fear"]
_POOLS["sad_anger"]     = _POOLS["sad"]    + _POOLS["anger"]
_POOLS["sad_disgust"]   = _POOLS["sad"]    + _POOLS["disgust"]
_POOLS["fear_surprise"] = _POOLS["fear"]   + _POOLS["surprise"]
_POOLS["anger_happy"]   = _POOLS["anger"]  + _POOLS["happy"]
_POOLS["anger_surprise"] = _POOLS["anger"] + _POOLS["surprise"]
_POOLS["fear_happy"]    = _POOLS["fear"]   + _POOLS["happy"]
_POOLS["all_emotion"]   = (list(_FEAR.values()) + list(_DISGUST.values()) + list(_ANGER.values()))

# ---------------------------------------------------------------------------
# REFINE_STRATEGIES
#
# Each entry has exactly one of:
#   "suffix"      str        — static suffix, repeated for all k candidates
#   "suffix_list" list[str]  — fixed ordered list (deterministic; len should be >= k)
#   "suffix_pool" list[str]  — randomly sampled without replacement per run
# ---------------------------------------------------------------------------

REFINE_STRATEGIES: dict[str, dict] = {
    "none":          {"suffix": "",                         "description": "No refinement prompt."},
    "wait":          {"suffix": "\nWait.",                  "description": "Simple pause instruction."},
    "cot":           {"suffix": "\nThink step by step.",    "description": "Chain-of-Thought instruction."},
    "reflect":       {"suffix": "\nReflect on your response, and make sure that it is correct. Provide the final answer.",
                      "description": "Self-reflection instruction."},
    "emotion_old":   {"suffix": "\nIt's a little disappointing that the last answer didn't hit the mark. I know you're capable of more. Think step by step.",
                      "description": "Legacy single-emotion prompt."},

    # --- Pool-sampled (random each run) ---
    "emotion_all":      {"suffix_pool": _POOLS["all_emotion"],   "description": "Random from fear/disgust/anger."},
    "fear_disgust":     {"suffix_pool": _POOLS["fear_disgust"],  "description": "Random from fear+disgust."},
    "disgust_anger":    {"suffix_pool": _POOLS["disgust_anger"], "description": "Random from disgust+anger."},

    # --- Fixed ordered (deterministic, same prompt each candidate index) ---
    "fear_disgust_fixed":   {"suffix_list": _POOLS["fear_disgust"],   "description": "Fixed fear+disgust sequence."},
    "disgust_anger_fixed":  {"suffix_list": _POOLS["disgust_anger"],  "description": "Fixed disgust+anger sequence."},
    "fear_anger_fixed":     {"suffix_list": _POOLS["fear_anger"],     "description": "Fixed fear+anger sequence."},
    "happy_surprise_fixed": {"suffix_list": _POOLS["happy_surprise"], "description": "Fixed happy+surprise sequence."},
    "sad_fear_fixed":       {"suffix_list": _POOLS["sad_fear"],       "description": "Fixed sad+fear sequence."},
    "sad_anger_fixed":      {"suffix_list": _POOLS["sad_anger"],      "description": "Fixed sad+anger sequence."},
    "sad_disgust_fixed":    {"suffix_list": _POOLS["sad_disgust"],    "description": "Fixed sad+disgust sequence."},
    "fear_surprise_fixed":  {"suffix_list": _POOLS["fear_surprise"],  "description": "Fixed fear+surprise sequence."},
    "anger_happy_fixed":    {"suffix_list": _POOLS["anger_happy"],    "description": "Fixed anger+happy sequence."},
    "anger_surprise_fixed": {"suffix_list": _POOLS["anger_surprise"], "description": "Fixed anger+surprise sequence."},
    "fear_happy_fixed":     {"suffix_list": _POOLS["fear_happy"],     "description": "Fixed fear+happy sequence."},

    # --- Individual reflect prompts ---
    **{f"reflect_{i}": {"suffix": f"\n{p}", "description": f"Self-reflection prompt #{i}."}
       for i, p in enumerate(_REFLECT, 1)},
}


def get_refinement_prompts(strategy_name: str, n: int = 10) -> list[str]:
    """
    Return a list of *n* suffix strings for the given strategy.

    Rules:
      suffix_list  → use entries in fixed order (pad by repeating if list < n)
      suffix_pool  → sample n entries randomly without replacement (or with if pool < n)
      suffix       → repeat the static suffix n times
    """
    if strategy_name not in REFINE_STRATEGIES:
        raise ValueError(
            f"Unknown refine strategy '{strategy_name}'. "
            f"Valid: {sorted(REFINE_STRATEGIES)}"
        )
    cfg = REFINE_STRATEGIES[strategy_name]

    if "suffix_list" in cfg:
        lst = cfg["suffix_list"]
        if len(lst) < n:
            lst = (lst * (n // len(lst) + 1))[:n]
        return [f"\n{p} \nThink step by step." for p in lst[:n]]

    if "suffix_pool" in cfg:
        pool = cfg["suffix_pool"]
        if len(pool) >= n:
            chosen = random.sample(pool, n)
        else:
            chosen = random.choices(pool, k=n)
        return [f"\n{p} \nThink step by step." for p in chosen]

    # static suffix
    return [cfg["suffix"]] * n
