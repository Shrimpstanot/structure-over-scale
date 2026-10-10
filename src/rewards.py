import re
import json

def _extract_last_boxed_content(text: str) -> str | None:
    """
    Robustly extracts the content of the LAST \boxed{...} tag.
    Handles nested braces correctly (e.g., \boxed{\frac{1}{2}}).
    Returns None if no boxed tag is found.
    """
    if not isinstance(text, str):
        return None
        
    start_idx = text.rfind("\\boxed{")
    if start_idx == -1:
        return None
    
    content_start = start_idx + 7  # len("\\boxed{") is 7
    
    balance = 0
    for i in range(content_start, len(text)):
        char = text[i]
        if char == "{":
            balance += 1
        elif char == "}":
            if balance == 0:
                return text[content_start:i]
            balance -= 1
            
    return None


def skywork_elite_transform(cfg, *args, **kwargs):
    """
    Transform for the Skywork Elite Checkpoint B dataset.
    - Parses the 'prompt' column (list of dicts) into a string.
    - Parses the 'reward_model' column to extract and clean the ground truth.
    - Formats for R1-style reasoning with <think>.
    """

    def transform_fn(example, tokenizer=None):
        raw_prompt = example.get("prompt", [])
        q_text = ""
        
        if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
            q_text = raw_prompt[0].get("content", "")
        elif isinstance(raw_prompt, str):
            q_text = raw_prompt

        raw_reward = example.get("reward_model", {})
        f_answer = ""
        
        if isinstance(raw_reward, str):
            try:
                raw_reward = json.loads(raw_reward)
            except (json.JSONDecodeError, TypeError):
                raw_reward = {}

        gt_value = raw_reward.get("ground_truth", "")
        
        if isinstance(gt_value, str):
            try:
                parsed_gt = json.loads(gt_value)
                
                if isinstance(parsed_gt, list) and len(parsed_gt) > 0:
                    f_answer = str(parsed_gt[0])
                else:
                    f_answer = str(parsed_gt)
            except (json.JSONDecodeError, TypeError):
                f_answer = gt_value.strip('[]"\'')
        else:
            f_answer = str(gt_value)

        prompt_text = (
            f"A conversation between a user and a helpful AI assistant.\n"
            f"User: {q_text}\n"
            f"Assistant: <think>\n"
        )
        
        return {
            "prompt": prompt_text,
            "solution": f_answer,
        }

    return transform_fn, {
        "remove_columns": [
            "prompt", "reward_model", "extra_info", "data_source", "ability"
        ]
    }


def math_dataset_transform(cfg, *args, **kwargs):
    """
    Transform for the new MATH/Algebra dataset.
    Extracts 'problem' and parses the last \boxed{} from 'solution'.
    """
    
    def transform_fn(example, tokenizer=None):
        q_text = example.get("problem", "")
        raw_sol = example.get("solution", "")
        
        clean_sol = _extract_last_boxed_content(raw_sol)
        
        if clean_sol is None:
            if raw_sol.startswith("$") and raw_sol.endswith("$"):
                clean_sol = raw_sol[1:-1]
            else:
                clean_sol = raw_sol

        prompt_text = (
            f"A conversation between a user and a helpful AI assistant.\n"
            f"User: {q_text}\n"
            f"Assistant: <think>\n"
        )
        
        return {
            "prompt": prompt_text,
            "solution": clean_sol,
        }

    return transform_fn, {"remove_columns": ["problem", "solution", "level", "type"]}



def deepmath_transform(cfg, *args, **kwargs):
    """
    Transform for the DeepMath-103K dataset.
    Filters for 'Hardcore' difficulty (8.5 - 10.0) only.
    """

    def transform_fn(example, tokenizer=None):
        q_text = example.get("question", "")
        f_answer = example.get("final_answer", "")

        prompt_text = (
            f"A conversation between a user and a helpful AI assistant.\n"
            f"User: {q_text}\n"
            f"Assistant: <think>\n"
        )
        
        return {
            "prompt": prompt_text,
            "solution": f_answer, 
        }

    return transform_fn, {
        "remove_columns": [
            "question", "final_answer", "difficulty", "topic", 
            "r1_solution_1", "r1_solution_2", "r1_solution_3"
        ]
    }


try:
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, NormalizationConfig
except ImportError as exc:
    raise ImportError(
        "math_verify is required for reward computation. Install requirements.txt first."
    ) from exc


def _get_content_text(completion) -> str:
    """
    Safely extracts text content regardless of whether TRL passes 
    a raw string, a list of messages, or a dictionary.
    """
    if isinstance(completion, str):
        return completion
        
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
        
    if isinstance(completion, list) and len(completion) > 0:
        last_item = completion[-1]
        if isinstance(last_item, dict):
            return str(last_item.get("content", ""))
        return str(last_item)
        
    return str(completion)


def gsm8k_solution_transform(cfg, *args, **kwargs):
    """
    FACTORY FUNCTION: This is what Axolotl calls first.
    Returns the transform_fn and dataset args.
    """
    
    def transform_fn(example, tokenizer=None):
        raw_answer = example.get("answer", "")
        sol = raw_answer.split("####")[-1].strip()

        q_text = example.get("question", "")
        
        prompt_text = (
            f"A conversation between a user and a helpful AI assistant.\n"
            f"User: {q_text}\n"
            f"Assistant: <think>\n"
        )
        return {
            "prompt": prompt_text,
            "solution": sol,
        }

    return transform_fn, {"remove_columns": ["question", "answer"]}

def gsm8k_solution_transform_no_think(cfg, *args, **kwargs):
    """
    FACTORY FUNCTION: This is what Axolotl calls first.
    Returns the transform_fn and dataset args.
    """
    
    def transform_fn(example, tokenizer=None):
        raw_answer = example.get("answer", "")
        sol = raw_answer.split("####")[-1].strip()

        q_text = example.get("question", "")
        
        prompt_text = (
            f"A conversation between a user and a helpful AI assistant.\n"
            f"User: {q_text}\n"
            f"Assistant: \n"
        )
        return {
            "prompt": prompt_text,
            "solution": sol,
        }

    return transform_fn, {"remove_columns": ["question", "answer"]}

def accuracy_r(completions, solution, **kwargs) -> list[float]:
    """
    Accuracy reward using math_verify.
    Uses _get_content_text to be safe.
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        content = _get_content_text(completion)
        
        if "</think>" in content:
            search_text = content.split("</think>")[-1]
        else:
            search_text = content
        
        gold_parsed = parse(
            f"${sol}$",
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, try_extract_without_anchor=True
                )
            ],
        )

        if len(gold_parsed) != 0:
            answer_parsed = parse(
                search_text,
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        normalization_config=NormalizationConfig(
                            basic_latex=True,
                            units=True,
                            malformed_operators=False,
                            nits=False,
                            boxed=True,
                        ),
                        try_extract_without_anchor=True,
                    )
                ],
                extraction_mode="last_match",
            )

            try:
                val = verify(gold_parsed, answer_parsed)
                reward = 1.0 if val else 0.0
            except Exception:
                reward = 0.0
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards


def strict_format_reward_shaped_4_empty(completions, **kwargs) -> list[float]:
    """
    Rewards the expected think-tag structure and non-empty reasoning.
    """
    rewards = []
    turn_pattern = r"(?i)(?:^|\n)\s*(User|Assistant):"

    for completion in completions:
        content = _get_content_text(completion)
        score = 0.0
        
        if re.search(turn_pattern, content):
            rewards.append(-0.75)
            continue

        open_tags = content.count("<think>")
        close_tags = content.count("</think>")
        
        if close_tags == 1:
            score += 0.375
        elif close_tags > 1:
            score -= 0.1
        else:
            score -= 0.3
        
        if open_tags > 0:
            score -= 0.2
        
        if "</think>" in content and close_tags == 1:
            thought_content = content.split("</think>")[0]
            thought_content = thought_content.replace("<think>", "").strip()
            words = re.findall(r'\w+', thought_content)
            
            if len(words) == 0: 
                score -= 0.6
                rewards.append(score)
                continue
            
            lines = [l.strip().lower() for l in content.split('\n') if len(l.strip()) > 3]
            if lines:
                most_common_line_count = max(lines.count(l) for l in set(lines))
                if most_common_line_count > 3:
                    score -= 0.5
                    rewards.append(score)
                    continue
                     
            words = re.findall(r'\w+', content.lower())
            n = 3
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            if ngrams:
                unique_ngrams = set(ngrams)
                repetition_ratio = len(unique_ngrams) / len(ngrams)
                
                threshold = 0.5
                if repetition_ratio < threshold:
                    score -= 0.4
            
            answer_content = content.split("</think>")[1]
            if len(answer_content) > 0:
                score += 0.375

        rewards.append(score)
        
    return rewards

