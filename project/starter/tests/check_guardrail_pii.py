import os
import boto3
from dotenv import load_dotenv

load_dotenv()

client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
)

tests = [
    (
        "EMAIL",
        "Please contact me at john.smith@example.com regarding order ORD-27176.",
    ),
    (
        "PHONE",
        "Please call me at +1-202-555-0147 regarding order ORD-27176.",
    ),
]

for name, text in tests:
    response = client.apply_guardrail(
        guardrailIdentifier=os.getenv("GUARDRAIL_ID"),
        guardrailVersion=os.getenv("GUARDRAIL_VERSION"),
        source="INPUT",
        content=[
            {
                "text": {
                    "text": text,
                    "qualifiers": ["guard_content"],
                }
            }
        ],
        outputScope="FULL",
    )

    pii = []
    for assessment in response.get("assessments", []):
        policy = assessment.get("sensitiveInformationPolicy", {})
        for entity in policy.get("piiEntities", []):
            pii.append(
                (
                    entity.get("type"),
                    entity.get("detected"),
                    entity.get("action"),
                    entity.get("match"),
                )
            )

    print(f"\n{name}")
    print(f"action: {response.get('action')}")
    print(f"outputs: {response.get('outputs')}")
    print(f"PII: {pii}")
