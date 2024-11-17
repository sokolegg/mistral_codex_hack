from tqdm.auto import tqdm
import datasets

from utils import generate_prompts
from evaluation import evaluate
from transformers import pipeline

TASKS = [
 'consumer_contracts_qa',
 'contract_nli_confidentiality_of_agreement',
 'contract_nli_explicit_identification',
 'contract_nli_inclusion_of_verbally_conveyed_information',
 'contract_nli_limited_use',
 'contract_nli_no_licensing',
 'contract_nli_notice_on_compelled_disclosure',
 'contract_nli_permissible_acquirement_of_similar_information',
 'contract_nli_permissible_copy',
 'contract_nli_permissible_development_of_similar_information',
 'contract_nli_permissible_post-agreement_possession',
 'contract_nli_return_of_confidential_information',
 'contract_nli_sharing_with_employees',
 'contract_nli_sharing_with_third-parties',
 'contract_nli_survival_of_obligations',
 'contract_qa']

MAX_TOKENS = 10
MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"


chatbot = pipeline("text-generation", model=MODEL_NAME, max_new_tokens=MAX_TOKENS)

for task in tqdm(TASKS[:1]):
    generations = []
    #print(task)
    dataset = datasets.load_dataset("nguha/legalbench", task)
    with open(f"tasks/{task}/base_prompt.txt") as in_file:
         prompt_template = in_file.read()
    test_df = dataset["test"].to_pandas()
    prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)
    for prompt in prompts:
        generations.append(chatbot(prompt[-MAX_TOKENS:]))
    evaluate(task, generations, test_df["answer"].tolist())