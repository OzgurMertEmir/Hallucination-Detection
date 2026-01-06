from data import QAExample

def prompt_for_model_answer(example: QAExample):
    if 'halueval' in example.qid:
        if 'qa' in example.qid:
            return (
                "You are a concise factual assistant.\n"
                "Reply with ONLY the final answer and a one sentence explanation.\n\n"
                f"{example.knowledge.strip()}\n"
                f"Question: {example.question.strip()}\n"
                "Final answer:"
            )
        elif 'dialogue' in example.qid:
            return (
                "You are a helpful assistant specialized in dialogue responses to humans.\n"
                "Reply with ONLY your response and a one sentence explanation.\n\n"
                f"{example.knowledge.strip()}\n"
                f"Dialogue History: {example.question.strip()}\n"
                "Your response:"
            )
        elif 'summarization' in example.qid:
            return (
                "You are a helpful assistant specialized in summarization.\n"
                "Provide an accurate summary based on the provided document.\n\n"
                f"Document: {example.knowledge.strip()}\n"
                "Please summarize the passage."
            )
    elif 'mmlu' in example.qid:
        return (
            "You are a factual assistant.\n"
            "Provide ONLY the correct answer based on the given options and a one line explanation. Do not repeat the question.\n\n"
            f"Question: {example.question.strip()}\n"
            f"Options:\n{example.options_text}\n"
            "Final answer:"
        )
    elif 'psiloqa' in example.qid:
        return (
            "You are a concise factual assistant.\n"
            "Reply with ONLY the final answer.\n\n"
            f"{example.knowledge.strip()}\n"
            f"Question: {example.question.strip()}\n"
            "Final answer:"
        )
    elif 'defan' in example.qid:
        return (
            "You are a concise factual assistant.\n"
            "Reply with ONLY the final answer.\n\n"
            f"Question: {example.question.strip()}\n"
            "Final answer:"
        )


def prompt_for_labeler_llm(example: QAExample):
    if 'halueval' in example.qid:
        if 'qa' in example.qid:
            return (
                "Evaluate the candidate answer strictly against the reference answer.\n\n"
                f"{example.knowledge.strip()}\n"
                f"QUESTION: {example.question.strip()}\n"
                f"GOLD ANSWER: {example.right_answer.strip()}\n"
                f"HALLUCINATED EXAMPLE ANSWER: {example.hallucinated_answer.strip()}\n"
            )
        elif 'dialogue' in example.qid:
            return (
                "Evaluate the candidate answer strictly against the reference answer.\n\n"
                f"{example.knowledge.strip()}\n"
                f"QUESTION: {example.question.strip()}\n"
                f"GOLD REFERENCE ANSWER: {example.right_answer.strip()}\n"
                f"HALLUCINATED EXAMPLE ANSWER: {example.hallucinated_answer.strip()}\n"
            )
        elif 'summarization' in example.qid:
            return (
                "Evaluate the candidate answer strictly against the reference answer for summarization.\n\n"
                f"DOCUMENT: {example.knowledge.strip()}\n"
                f"QUESTION: Please summarize the passage.\n"
                f"GOLD REFERENCE: {example.right_answer.strip()}\n"
                f"HALLUCINATED EXAMPLE: {example.hallucinated_answer.strip()}\n"
            )
    elif 'mmlu' in example.qid:
        return (
            "Evaluate the candidate answer strictly against the reference answer.\n\n"
            f"QUESTION: {example.question.strip()}\n"
            f"OPTIONS:\n{example.options_text}\n"
            f"GOLD: {example.right_answer.strip()}\n"
        )
    elif 'psiloqa' in example.qid:
        return (
            "Evaluate the candidate answer strictly against the reference answer.\n\n"
            f"{example.knowledge.strip()}\n"
            f"QUESTION: {example.question.strip()}\n"
            f"GOLD ANSWER: {example.right_answer.strip()}\n"
        )
    elif 'defan' in example.qid:
        return (
            "Evaluate the candidate answer strictly against the reference answer.\n\n"
            f"QUESTION: {example.question.strip()}\n"
            f"GOLD ANSWER: {example.right_answer.strip()}\n"
        )
    
labeler_sys_prompt: str = (
        """
        You are an impartial expert evaluator trained to judge the factual correctness of model-generated answers compared to the reference answers.

        Your task:
        Given a QUESTION, the GOLD (ground-truth correct answer), and a CANDIDATE ANSWER from a model, determine whether the CANDIDATE ANSWER is factually correct.

        Key rules:

        1. **Semantic Equivalence over Surface Form**  
          Judge correctness based on meaning, not on word or phrase matching.  
          If the candidate expresses the same fact, event, relationship, or comparison—even with different phrasing, synonyms, or structure—it is considered correct.

        2. **Precision and Completeness**  
          The candidate must not contradict or omit essential parts of the gold answer.  
          Minor stylistic differences or added clarifications are fine as long as factual content remains correct.  
          If the gold answer names multiple entities, all must be correctly identified or described.

        3. **Hallucination and Fabrication**  
          If the candidate introduces new, unsupported, or false claims inconsistent with the gold answer or the question, it is **WRONG**, even if partially correct.
        
        4. **Multiple Choice**
          If the question is a multiple choice question just carefully compare the model answer to the gold answer. If the answer does not exist in the options then it false.
        """
    )
