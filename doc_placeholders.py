import os
from mistralai import Mistral
import json
api_key = "5ZrznqDpDW1v3363Ksvps3XxoWh8Iwfy"
model = "pixtral-12b-2409"

client = Mistral(api_key=api_key)

docs_images = [
    "https://signaturely.com/wp-content/uploads/2022/08/non-disclosure-agreement-uplead.jpg", # Non-Disclosure Agreement
    "https://signaturely.com/wp-content/uploads/2022/08/work-for-hire-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/referral-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/property-management-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/agency-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/payment-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/non-compete-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/memorandum-of-understanding-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/loan-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/intellectual-property-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/employment-contract-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/rental-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/sublease-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/sales-contract-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/retainer-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/independent-contract-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/commission-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/service-agreement-uplead.jpg",
    "https://signaturely.com/wp-content/uploads/2022/08/video-release-uplead.jpg",
]

import tqdm

if __name__ == "__main__":
    model = "pixtral-12b-2409"

    doc_templates = []

    for doc_url in tqdm.tqdm(docs_images[:]):
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                        Parse this picture and return a structured result in text format.

                        """
                        },
                        {
                            "type": "image_url",
                            "image_url": f"{doc_url}"
                        }
                    ]
                }
            ]
        )

        doc_templates.append(chat_response.choices[0].message.content)

    docs_with_template = [
        {"doc_url": doc_url, "doc_template": doc_template}
        for doc_url, doc_template in zip(docs_images[:], doc_templates[:])
    ]

    with open("./data/docs_with_template.json", "w") as file:
        json.dump(docs_with_template, file, indent=4)
