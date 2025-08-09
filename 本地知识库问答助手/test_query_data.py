from query_data import query_rag
from get_embedding_function import get_embedding_function

from langchain_ollama import OllamaLLM
from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import load_evaluator

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response math the expected response?
"""

def test_sustainability_docs11():
    assert query_and_validate(
        question='What is the purpose of the trading agent described in the document?',
        expected_response='The purpose of the trading agent is to autonomously participate in a market or trading environment by making buy and sell decisions on behalf of a user or organization.'
    )

def query_and_validate(question:str, expected_response:str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response= response_text
    )

    model = OllamaLLM(model='qwen2.5:0.5b')
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()
    
    print(prompt)

    evaluator = load_evaluator("embedding_distance", embeddings=get_embedding_function(), distance_metric=EmbeddingDistance.EUCLIDEAN)

    if "true" in evaluation_results_str_cleaned:
        # print response in Green if it is correct
        print("\033[92m" + f"Response:{evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # get the pairwise distance between the expected result and actual result embeddings
        prob_result = evaluator.evaluate_strings(
            prediction=response_text, reference=expected_response
        )
        if prob_result['score'] >= 0.8:
            print("\033[92m" + f"Response:{evaluation_results_str_cleaned}" + "\033[0m")
            return True
        else:
            # Print response in red if it is incorrect
            print("\033[91m" + f"Response:{evaluation_results_str_cleaned}" + "\033[0m")
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )


