from mistralai import Mistral
import os
from typing import List, Dict, Any
import numpy as np
import tqdm
import re
import json
import dotenv
dotenv.load_dotenv()

# api_keys = [os.environ["MISTRAL_API_KEY"], os.environ["MISTRAL_API_KEY2"], os.environ["MISTRAL_API_KEY3"]]
# api_key = np.random.choice(api_keys, size=1)[0]


mistakes = [
    "The name for a certain personne CANT be different",
    "Dates related to one entity must match",
    "The address for a certain personne CANT be different. But if these are the addresses for human and compnay - it's not a mistake",
    "Inaccurate wording in the text",
    "The meaning changes as the text progresses",
    'Ambiguous Language',
    'Inconsistent Terminology',
    "Lack of Clarity: Using vague or unclear terms that lead to multiple interpretations (e.g., 'reasonable time' without defining it).",
    "Undefined Terms: Failing to define legal or technical jargon, especially terms specific to the case.",
    "Misuse of Pronouns: Overusing 'it,' 'they,' or 'he/she,' leading to confusion about which party or entity is being referred to.",

    "Using different terms for the same concept (e.g., 'party,' 'entity,' 'individual') without specifying they are interchangeable.",
    "Switching between active and passive voice inconsistently.",
    "Disorganized Layout: Failing to use headers, numbered clauses, or bullet points for readability and logical flow.",
    "Lack of Consistency in Formatting: Inconsistent font sizes, spacing, or indentation detracts from professionalism and readability.",
    "Incorrect Citations: Failing to cite relevant laws, statutes, or precedents correctly.",
    "Broken Cross-References: Linking to the wrong sections or documents, especially in lengthy contracts or briefs.",

    # "Even minor errors (e.g., misplaced commas) can change legal meanings.",
    # "Typos, especially in names, dates, or figures, can invalidate agreements or require corrections.",
    # "Leaving key terms or obligations undefined, which can lead to disputes (e.g., not specifying payment deadlines or deliverable standards).",
    # "Using broad or general terms that are inappropriate for the context.",
    # "Outdated Laws or Precedents: Citing repealed or amended statutes.",
    # "Failing to incorporate the latest case law or legislative changes.",
    # "Leaving out necessary clauses (e.g., force majeure, dispute resolution).",
    # "Including redundant clauses that add bulk without value.",
    # "Not tailoring the document to comply with the legal framework of the relevant jurisdiction.",
    # "Using templates designed for other jurisdictions without modification.",
    # "Blindly copying templates or precedent documents without tailoring them to the specific case.",
    # "Including irrelevant clauses that could harm the document’s enforceability.",
    # "Overlooking potential ambiguities or scenarios that might lead to conflict.",
    # "Drafting terms without considering how they will be enforced or interpreted in court.",
    # "Including confidential or sensitive information unnecessarily.",
    # "Violating attorney-client privilege by revealing privileged communications.",
    # "Failing to proofread or seek peer/legal review.",
    # "Rushing the finalization process, leading to easily avoidable mistakes.",
    # "Incorrectly naming parties (e.g., using informal names instead of legal entities).",
    # "Missing required signatures or authorization.",
]

# def invoke(model_name: str, template: str, mistakes: List[str]) -> Dict[str, Any]:
#     mistakes_str = "\n".join(mistakes)
#     chat_response = client.chat.complete(
#         model=model_name,
#         messages=[
#             {
#                 "role": "system",
#                 "content": f"""
#                 You are a perfect lawyer. You can create the amazing and clean document.
#
#                 You must follow next instructions:
#                 1. read a template
#                 2. imagine the information that you can paste in the gaps
#                 3. imagine additional information and extend the current template.
#                 Write a document after <document placeholder> placeholder
#                 4. make mistakes in the document based on the following list:
#                 Mistakes:
#                 {mistakes_str}
#                 5. after <mistakes placeholder> write explanation for mistakes that you did.
#
#                 Dont repeat mistakes. You should explain it.
#                 Dont return a template in the response.
#                 Do not offer solutions for these mistakes.
#
#                 Template:
#                 {template}
#                 """,
#             },
#             {
#                 "role": "user",
#                 "content": f"""
#                 <document placeholder>
#
#                 <mistakes placeholder>
#                 """,
#             },
#         ]
#     )
#     response = chat_response.choices[0].message.content
#     try:
#         mistake_pos_ = list(re.finditer("<mistakes placeholder>", response))[0].span()[-1]
#         obj = [
#             {
#                 "role": "user",
#                 "content": (
#                     response[:mistake_pos_]
#                     .replace("<document placeholder>", "")
#                     .replace('<mistakes placeholder>', '')
#                 ),
#             },
#             {
#                 "role": "assistant",
#                 "content": response[mistake_pos_:],
#             }
#         ]
#         return {"obj": obj, "mistakes": mistakes}
#     except Exception as ex:
#         print(ex)


def invoke(client, model_name: str, template: str, mistakes: List[str]) -> Dict[
    str, Any]:
    mistakes_str = "\n".join(mistakes)
    chat_response = client.chat.complete(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a lawyer with extensive experience.
                You have worked with various documents. You have resolved many controversial issues in your career.
                Now you must examine students who are studying at the Faculty of Law.

                You MUST create different names.
                You MUST create Different addresses.
                You MUST create Different companies.

                You must create a document using a template
                Template:
                {template}

                You must make the following mistakes in this text. 
                Mistakes:
                {mistakes_str}

                You must follow next instructions:
                1. read a template
                2. imagine the information that you can paste in the gaps
                3. imagine additional information and extend the current template.
                Write a document after <document placeholder> placeholder
                4. make mistakes in the document based on the following list:
                Mistakes:
                {mistakes_str}
                5. after <mistakes placeholder> write what mistakes you made and how students can fix it.

                Dont write words about students.
                Dont repeat mistakes. You should explain it.
                Dont return a template in the response.
                Do not offer solutions for these mistakes.
                """,
            },
            {
                "role": "user",
                "content": f"""
                <document placeholder>

                <mistakes placeholder>
                """,
            },
        ]
    )
    response = chat_response.choices[0].message.content
    try:
        mistake_pos_ = \
        list(re.finditer("<mistakes placeholder>", response))[0].span()[-1]
        obj = [
            {
                "role": "user",
                "content": (
                    response[:mistake_pos_]
                    .replace("<document placeholder>", "")
                    .replace('<mistakes placeholder>', '')
                ),
            },
            {
                "role": "assistant",
                "content": response[mistake_pos_:],
            }
        ]
        return {"obj": obj, "mistakes": mistakes}
    except Exception as ex:
        print(ex)


import click

@click.command()
@click.option('--api-key', default=str)
def run(api_key: str):
    for i in tqdm.tqdm(range(250)):
        client = Mistral(api_key=api_key)
        with open("./data/docs_with_template.json", "r") as file:
            docs_with_template = json.load(file)

        ix = np.random.choice(len(docs_with_template), size=1)[0]
        rnd_mistakes = list(np.random.choice(
            mistakes,
            size=min(np.random.choice(range(len(mistakes))), 10), replace=False
        ))
        incorrect_doc = invoke(
            client,
            "mistral-large-latest",
            docs_with_template[ix]["doc_template"],
            rnd_mistakes,
        )

        id = "".join(map(str, np.random.choice(range(100), size=15, replace=True)))
        with open(f"./data/{id}.json", "w") as file:
            json.dump(incorrect_doc, file, indent=4)


if __name__ == "__main__":
    run()
