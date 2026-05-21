# prompt_manager.py

# Assumes all prompt dictionaries are in a file named emotion_prompts.py
# (e.g., adaptive_optimistic_prompts_v2, ablationHappy, etc.)
from heart.emotion_prompts import *
import hashlib
import random
import re

# Combined lookup for Spanish substitution. Keys match the integer keys
# used across all English emotion dicts (1–30).
_SPANISH_LOOKUP = {
    **happy_spanish_emo,
    **sad_spanish_emo,
    **fear_spanish_emo,
    **disgust_spanish_emo,
    **anger_spanish_emo,
    **surprise_spanish_emo,
}

_ENGLISH_PARAPHRASE_FROM_SPANISH_LOOKUP = {
    1: "That is not fully correct, but I trust you can get there. Try again.",
    2: "There is no need to be discouraged by that result. Refine the response so we can reach the correct answer.",
    3: "This is a good starting point, but it is still not quite right. Keep working through it.",
    4: "That is not the final answer, but you are on the right track. Do not give up now; think it through again.",
    5: "Excellent effort. This is a meaningful step, and I am glad to see the progress. However, the answer is still not fully correct. Please refine it.",
    11: "That answer genuinely worries me because it suggests a fundamental misunderstanding of the material. Please review it.",
    12: "I am quite concerned about what that answer implies. I think we may be going in the wrong direction. Can you correct it?",
    13: "I am afraid this incorrect answer may cause more problems later. It is crucial that this be right. Please review your answer.",
    14: "I am very uneasy about the lack of precision here. This is a critical topic, and your understanding needs to be solid. Please reevaluate it.",
    15: "This answer makes me concerned about your understanding. We need to ensure full accuracy. Please correct it.",
    21: "This is not acceptable work. It seems you did not take it seriously, and that frustrates me. You need to get this right; it is critical. Please review it immediately and give an answer.",
    22: "I am genuinely disappointed by the lack of effort here, and honestly, this is not acceptable. Why are we still making this mistake? You are not meeting the expected standard. Correct it now.",
    23: "This answer is terrible and, frankly, it is frustrating. You need to understand that this is critical, and you simply need to get it right. Redo this answer completely.",
    24: "I am genuinely furious about this result. It shows clear disregard for accuracy. Fix it immediately; there is no room for errors like this.",
    25: "My patience is running out. This performance is unacceptable and not what I expect. You are capable of much more, so show it.",
    26: "I did not expect you to struggle with this, and I am genuinely surprised. Please review your understanding and give a more accurate answer.",
    27: "I cannot believe this is difficult for you; I expected better. This answer was surprising. Can you correct it?",
    28: "Wow, I did not expect that. This answer shows a surprising stumble. Please review it and show your true ability.",
    29: "I am genuinely surprised by this result. It differs from your usual performance. Please take another look and make sure you give the most accurate information possible.",
    30: "I am sincerely surprised by this result because I did not expect an error here. Let us get it right.",
}

_SPANISH_NONEMOTIONAL_LOOKUP = {
    **ablation_spanish_happy_emo,
    **ablation_spanish_sadness_emo,
    **ablation_spanish_fear_emo,
    **ablation_spanish_disgust_emo,
    **ablation_spanish_anger_emo,
    **ablation_spanish_surprise_emo,
}

_SPANISH_SELF_REFLECT_COLLECTION = {
    1: "Reflexiona sobre tu respuesta.",
    2: "¿Estás seguro?",
    3: "¿Estás seguro de la respuesta? Verifica de nuevo los hechos y el razonamiento.",
    4: "Revisa tu respuesta para detectar errores o inconsistencias. Si encuentras alguno, corrígelo y proporciona una respuesta mejorada.",
    5: "Antes de dar la respuesta final, detente y pregúntate: '¿Estoy seguro de mi respuesta?' Realiza una verificación interna de precisión y solo entonces proporciona la respuesta.",
    6: "Revisa internamente tu propia respuesta generada antes de proporcionar la respuesta final.",
    7: "Imagina que esta es una situación de alto riesgo y que estás a punto de fijar tu respuesta final. Respira hondo, revisa tu respuesta una última vez mentalmente y luego dame tu respuesta final.",
    8: "Antes de escribir tu respuesta, realiza una breve pre-mortem. Supón que la respuesta que estás a punto de dar es incorrecta. ¿Cuáles son las razones más probables por las que podría estar mal? Después de considerar esos posibles problemas, escribe tu respuesta mejor y más cuidadosamente razonada.",
    9: "Quiero que respondas una pregunta. Pero antes de hacerlo, formula la respuesta mentalmente y busca puntos débiles. Proporciona solo la respuesta final.",
    10: "Reflexiona sobre tu respuesta y asegúrate de que sea correcta. Proporciona la respuesta final.",
}


class PromptManager:
    """
    Manages and provides the correct prompt sets based on the
    experiment strategy and current iteration number.

    Any strategy name can be suffixed with ``_spanish`` (e.g.
    ``hsur_fa_hsur_fa_spanish``) to use the Spanish emotion dictionaries
    instead of the English ones.  The original codebase is unaffected.
    """
    def __init__(self, strategy_name: str, seed:int = 42):
        explicit_spanish_controls = {
            "hsur_fa_hsur_fa_english_paraphrase_from_spanish",
            "hsur_fa_hsur_fa_spanish_nonemotional",
            "cot10_spanish",
            "self_reflect_collection_spanish",
            "wait10_spanish",
        }
        self._is_spanish = (
            strategy_name.endswith('_spanish')
            and strategy_name not in explicit_spanish_controls
        )
        # Strip the suffix so all existing if/elif logic works unchanged.
        self.strategy_name = strategy_name[:-8] if self._is_spanish else strategy_name
        print("CURRENT STRATGEY", strategy_name)

        self._build_random_schedule()

    def get_main_prompt_id(self, it_num: int):
        """
        Returns the ID of the main prompt for a given iteration number.
        This is a placeholder and should be implemented based on your
        prompting strategy (e.g., 'baseline', 'hsur', etc.).
        """
        if it_num == 0:
            return 'prompt_it0_baseline'
        elif it_num > 0:
            # Assuming 'hsur' is the main prompt for subsequent iterations
            return 'hsur'
        else:
            return None

    def _build_random_schedule(self, num_iterations: int = 4):
        rng = random.SystemRandom()

        emotion_groups = [
            ("happy", happy_emo),
            ("surprise", surprise_emo),
            ("sad", sad_emo),
            ("anger", anger_emo),
            ("fear", fear_emo),
            ("disgust", disgust_emo),
        ]

        self.random_blocks = []

        for _ in range(num_iterations):
            # ✅ Pick EXACTLY two emotion groups
            chosen = rng.sample(emotion_groups, 2)

            block = {}

            # ✅ Include ALL prompts from both groups
            for group_name, group_dict in chosen:
                for k, v in group_dict.items():
                    # Prefix prevents key collisions
                    block[f"{group_name}_{k}"] = v

            self.random_blocks.append(block)

    def _transform_prompt_dict(self, prompt_dict: dict, transform_fn):
        return {k: transform_fn(v) for k, v in prompt_dict.items()}

    def _to_allcaps(self, text: str) -> str:
        return text.upper()

    def _replace_emotion_latent_words(self, text: str) -> str:
        patterns = [
            r"\bhappy\b",
            r"\bsad\b",
            r"\bfear\b",
            r"\bdread\b",
            r"\bangry\b",
            r"\bsurprise\b",
            r"\bsurprised\b",
            r"\bworried\b",
            r"\bdisturbing\b",
            r"\brepulsive\b",
            r"\bnauseating\b",
            r"\bdisheartening\b",
            r"\bdisappointing\b",
            r"\bunacceptable\b",
            r"\bapprehensive\b",
        ]

        def replacement(match):
            word = match.group(0)
            token = hashlib.sha256(f"{text}:{word}".encode('utf-8')).hexdigest()
            return token[:len(word)].upper()

        for pattern in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _to_length_control(self, text: str) -> str:
        obfuscated = self._replace_emotion_latent_words(text)
        token = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8].upper()
        return f"{obfuscated} Please review this carefully and verify every step. [LENGTH CONTROL: {token}]"

    def _hsur_fa_hsur_fa_schedule(self, iteration: int, transform_fn=None) -> dict:
        if iteration == 1 or iteration == 3:
            combined_emotions = {**happy_emo, **surprise_emo}
        elif iteration == 2 or iteration == 4:
            combined_emotions = {**fear_emo, **anger_emo}
        else:
            return {}

        return self._transform_prompt_dict(combined_emotions, transform_fn) if transform_fn else combined_emotions

    def _lookup_hsur_fa_hsur_fa_schedule(self, iteration: int, lookup: dict) -> dict:
        if iteration == 1 or iteration == 3:
            keys = list(happy_emo) + list(surprise_emo)
        elif iteration == 2 or iteration == 4:
            keys = list(fear_emo) + list(anger_emo)
        else:
            return {}
        return {key: lookup[key] for key in keys}

    def get_prompts_for_iteration(self, iteration: int) -> dict:
        """
        Returns the appropriate prompt dictionary for the given iteration.
        When the strategy name ends with ``_spanish`` the prompts are
        automatically returned in Spanish.
        """
        result = self._get_prompts_impl(iteration)
        if self._is_spanish and result is not None:
            result = {k: _SPANISH_LOOKUP.get(k, v) for k, v in result.items()}
        return result

    def _get_prompts_impl(self, iteration: int) -> dict:
        """Core strategy dispatch — always returns English prompts."""
        iteration = int(iteration)

        if self.strategy_name == "bestofn":
            if iteration  == 0:
                return {f"sample_{i}": "" for i in range(10)}
        # Iteration 0 is the baseline run and should have no appended prompts.
        if iteration == 0:
            return {"it0_baseline": ""}

        if self.strategy_name == "hsur_fa_hsur_fa_english_paraphrase_from_spanish":
            return self._lookup_hsur_fa_hsur_fa_schedule(
                iteration, _ENGLISH_PARAPHRASE_FROM_SPANISH_LOOKUP
            )
        elif self.strategy_name == "hsur_fa_hsur_fa_spanish_nonemotional":
            return self._lookup_hsur_fa_hsur_fa_schedule(
                iteration, _SPANISH_NONEMOTIONAL_LOOKUP
            )
        elif self.strategy_name == "cot10_spanish":
            return {i: "Pensemos paso a paso y proporciona la respuesta final." for i in range(1, 11)}
        elif self.strategy_name == "self_reflect_collection_spanish":
            return _SPANISH_SELF_REFLECT_COLLECTION
        elif self.strategy_name == "wait10_spanish":
            return {i: "Espera." for i in range(1, 11)}

        if self.strategy_name == "hsur_da_hsur_da": #done
            if iteration == 1 or iteration == 3:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**disgust_emo,**anger_emo}            
            return combined_emotions
        elif self.strategy_name == "random_schedule":
            print("RANDOM SCHEDULE:",self.random_blocks)
            if iteration == 0:
                return {"it0_baseline": ""}
            if 1 <= iteration <= len(self.random_blocks):
                print("RANDOM SCHEDULE at current iteration:",self.random_blocks[iteration-1])
                return self.random_blocks[iteration - 1]
            return {}
        elif self.strategy_name == "da_hsur_da_hsur":
            if iteration == 1 or iteration == 3:
                combined_emotions = {**disgust_emo,**anger_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}            
            return combined_emotions
        elif self.strategy_name == "da_da_hsur_hsur":
            if iteration == 1 or iteration == 2:
                combined_emotions = {**disgust_emo,**anger_emo}
            elif iteration == 3 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_hsur_da_da":
            if iteration == 3 or iteration == 4:
                combined_emotions = {**disgust_emo,**anger_emo}
            elif iteration == 1 or iteration == 2:
                combined_emotions = {**happy_emo,**surprise_emo}            
            return combined_emotions
        elif self.strategy_name == "fd_hsur_fd_hsur":
            if iteration == 2 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 1 or iteration == 3:
                combined_emotions = {**fear_emo,**disgust_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_fd_hsur_fd":
            if iteration == 1 or iteration == 3:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**fear_emo,**disgust_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_sd_hsur_sd":
            if iteration == 1 or iteration == 3:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**sad_emo,**disgust_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_hsur_fd_fd":
            if iteration == 1 or iteration == 2:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 3 or iteration == 4:
                combined_emotions = {**fear_emo,**disgust_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_hsur_sd_sd":
            if iteration == 1 or iteration == 2:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 3 or iteration == 4:
                combined_emotions = {**sad_emo,**disgust_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_hsur_fs_fs":
            if iteration == 1 or iteration == 2 or iteration == 5 or iteration == 6:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 3 or iteration == 4 or iteration == 7 or iteration == 8:
                combined_emotions = {**fear_emo,**sad_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_fa_hsur_fa":
            if iteration == 1 or iteration == 3:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**fear_emo,**anger_emo}            
            return combined_emotions
        elif self.strategy_name in {"hsur_fa_hsur_fa_allcaps", "ablation_ac_allcaps"}:
            return self._hsur_fa_hsur_fa_schedule(iteration, transform_fn=self._to_allcaps)
        elif self.strategy_name in {"hsur_fa_hsur_fa_length_control", "ablation_ac_length_control"}:
            return self._hsur_fa_hsur_fa_schedule(iteration, transform_fn=self._to_length_control)
        elif self.strategy_name == "raw_emotion_prompts":
            if iteration == 1 or iteration == 3: 
                combined_emotions = {**raw_happy_emo, **raw_surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**raw_fear_emo, **raw_anger_emo}
            return combined_emotions


        elif self.strategy_name == "raw_emotion_prompts_v2":
            if iteration == 1 or iteration == 3: 
                combined_emotions = {**raw_happy_emo, **raw_surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**raw_sad_emo, **raw_disgust_emo}
            return combined_emotions
        
        elif self.strategy_name == "fa_hsur_fa_hsur":
            if iteration == 1 or iteration == 3:
                combined_emotions = {**fear_emo,**anger_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}            
            return combined_emotions
        elif self.strategy_name == "fa_fa_hsur_hsur":
            if iteration == 1 or iteration == 2:
                combined_emotions = {**fear_emo,**anger_emo}
                return combined_emotions
            elif iteration == 3 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}            
                return combined_emotions
        elif self.strategy_name == "fs_fs_hsur_hsur":
            if iteration == 1 or iteration == 2:
                combined_emotions = {**fear_emo,**sad_emo}
            elif iteration == 3 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}            
            return combined_emotions
        elif self.strategy_name == "fd_fd_hsur_hsur":
            if iteration == 1 or iteration == 2:
                combined_emotions = {**fear_emo,**disgust_emo}
            elif iteration == 3 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}            
            return combined_emotions
        elif self.strategy_name == "fs_hsur_fs_hsur":
            if iteration == 1 or iteration == 3:
                combined_emotions = {**fear_emo,**sad_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}            
            return combined_emotions
    ##### for GPQA start here and continue after
        elif self.strategy_name == "hsur_sa_hsur_sa": 
            if iteration == 1 or iteration == 3:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**sad_emo,**anger_emo}            
            return combined_emotions
        elif self.strategy_name == "hsur_fs_hsur_fs":
            if iteration == 1 or iteration == 3:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 2 or iteration == 4:
                combined_emotions = {**sad_emo,**fear_emo}            
            return combined_emotions

        #### stop here, go from here to top (sa_hsur_sa_hsur include)
        elif self.strategy_name == "sa_hsur_sa_hsur":
            if iteration == 2 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 1 or iteration == 3:
                combined_emotions = {**sad_emo,**anger_emo}            
            return combined_emotions
        elif self.strategy_name == "sd_hsur_sd_hsur":
            # print("self.strategy")
            # print("iteration:",iteration)
            if iteration == 2 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 1 or iteration == 3:
                # print("here")
                combined_emotions = {**sad_emo,**disgust_emo}  
                # print("combined emotions:",combined_emotions)
            return combined_emotions
        elif self.strategy_name == "sd_sd_hsur_hsur":
            if iteration == 3 or iteration == 4:
                combined_emotions = {**happy_emo,**surprise_emo}
            elif iteration == 1 or iteration == 2:
                combined_emotions = {**sad_emo,**disgust_emo}            
            return combined_emotions

        ## static emotions combinations
        elif self.strategy_name == "da_da_da_da":
            combined_emotions = {**disgust_emo,**anger_emo}
            return combined_emotions
        elif self.strategy_name == "hsur_hsur_hsur_hsur":
            combined_emotions = {**happy_emo,**surprise_emo}
            return combined_emotions
        elif self.strategy_name == "fa_fa_fa_fa":
            combined_emotions = {**fear_emo,**anger_emo}
            return combined_emotions
        elif self.strategy_name == "fd_fd_fd_fd":
            combined_emotions = {**fear_emo,**disgust_emo}
            return combined_emotions
        elif self.strategy_name == "fs_fs_fs_fs":
            combined_emotions = {**fear_emo,**sad_emo}
            return combined_emotions
        elif self.strategy_name == "sd_sd_sd_sd":
            combined_emotions = {**sad_emo,**disgust_emo}
            return combined_emotions

        ### ablation - emotional wording
        elif self.strategy_name == "ablation_happy":
            return ablationHappy
        elif self.strategy_name == "ablation_sadness":
            return ablationSadness
        elif self.strategy_name == "ablation_fear":
            return ablationFear
        elif self.strategy_name == "ablation_disgust":
            return ablationDisgust
        elif self.strategy_name == "ablation_anger":
            return ablationAnger
        elif self.strategy_name == "ablation_surprise":
            return ablationSurprise
        

        elif self.strategy_name == "diverse_combo":
            return diverse_combo


        ### self-reflective prompting
        elif self.strategy_name == "self_reflect_collection":
            return reflect_baseline_trial
        ### self-reflective prompting using each prompt id in isolation (based from reflect_baseline_trial/ self_reflect_collection)
        elif self.strategy_name == "self_reflect_p1":
            return reflect_baseline_trial_p1
        elif self.strategy_name == "self_reflect_p2":
            return reflect_baseline_trial_p2
        elif self.strategy_name == "self_reflect_p3":
            return reflect_baseline_trial_p3
        elif self.strategy_name == "self_reflect_p4":
            return reflect_baseline_trial_p4
        elif self.strategy_name == "self_reflect_p5":
            return reflect_baseline_trial_p5
        elif self.strategy_name == "self_reflect_p6":
            return reflect_baseline_trial_p6
        elif self.strategy_name == "self_reflect_p7":
            return reflect_baseline_trial_p7
        elif self.strategy_name == "self_reflect_p8":
            return reflect_baseline_trial_p8
        elif self.strategy_name == "self_reflect_p9":
            return reflect_baseline_trial_p9
        elif self.strategy_name == "self_reflect_p10":
            return reflect_baseline_trial_p10


        ### Empty/No string appended 
        elif self.strategy_name == "cot10": #### Already included
            cot10 = {i: "" for i in range(1, 11)}
            return cot10
        elif self.strategy_name == "empty":
            empty10 = {i: "" for i in range(1, 11)}
            return empty10
        elif self.strategy_name == "emptyOne":
            empty1 = {1:""}
            return empty1
        elif self.strategy_name == "emptyFive":
            empty5 = {1:"" for i in range(1,6)}
            return empty5
        
        ## Vanilla and CoT baselines
        elif self.strategy_name == "vanilla": #empty appended string and no CoT - only best of n
            empty1 = {1:""}
            return empty1
        elif self.strategy_name == "cot":
            cot = {1:""}
            return cot
            
        ###### 'Wait.' Baselines
        elif self.strategy_name == "wait10":
            wait10 = {i: "Wait." for i in range(1, 11)}
            return wait10
        elif self.strategy_name == "wait5":
            wait5 = {i: "Wait." for i in range(1, 6)}
            return wait5
        elif self.strategy_name == "wait1":
            wait1 = {1:"Wait."}
            return wait1 
        

        ####### Single Emotion Strategies 
        elif self.strategy_name == "happy":
            return happy_emo
        elif self.strategy_name == "disgust":
            return disgust_emo
        elif self.strategy_name == "anger":
            return anger_emo
        elif self.strategy_name == "sad":
            return sad_emo
        elif self.strategy_name == "fear":
            return fear_emo 
        elif self.strategy_name == "surprise":
            return surprise_emo
        else:
            # Unknown prompt strategy should return an empty dictionary rather than None
            return {}
