from openai import AzureOpenAI
import os
from azure.identity import ManagedIdentityCredential
from dotenv import load_dotenv
import json

load_dotenv()

# Load environment variables
OPENAI_Endpoint = os.getenv("OPENAI_ENDPOINT")
OPENAI_Api_version = os.getenv("OPENAI_API_VERSION")
OPENAI_model_name = os.getenv("OPENAI_MODEL_NAME")
UAMI_CLIENT_ID = os.getenv("UAMI_CLIENT_ID")

# Managed Identity
credential = ManagedIdentityCredential(client_id=UAMI_CLIENT_ID)

# Predefined categories
CATEGORIES = [
    "Abandoned Vehicle", "Accessibility", "Animal - Deceased", "Animal - Domestic", "Animal - General",
    "Damaged Road", "Damaged Street Sign", "Dumped Rubbish", "Dumped Tyres", "Facility - General Request",
    "Fallen Tree", "General - Abandoned Trolley", "General Request", "Graffiti - General", "Graffiti - Public Property",
    "Graffiti - Signage", "Illegal Parking", "Litter", "Noise - Animal", "Noise - Construction", "Noise - General",
    "Overgrown Vegetation", "Park - General Request", "Parking - Disabled", "Pavement - Damaged", "Pavement - General",
    "Pest / Vermin", "Pit and Equipment - General", "Playground Equipment", "Poles and Signage - General",
    "Pollution - General", "Pothole", "Public Toilet", "Request Bin Repair or Replacement", "Road Blockage",
    "Road Signage", "Roads - General", "Rubbish and Bins - General", "Street Cleaning", "Street Gutters / Storm Water",
    "Trees - General", "Vandalism - General", "Water Fountain"
]


def DocProcessor(image_url):
    """Analyze the image using GPT-5 only and return captions + category, matching previous output format."""

    def provider():
        return credential.get_token("https://cognitiveservices.azure.com/.default").token

    # Initialize GPT client
    client = AzureOpenAI(
        api_version=OPENAI_Api_version,
        azure_endpoint=OPENAI_Endpoint,
        azure_ad_token_provider=provider,
    )

    # Build GPT prompt
    messages = [
        {
            "role": "system",
            "content": f"""
            You are a visual reasoning assistant. Analyze the image and respond ONLY with valid JSON in this format:

            {{
              "main_caption": "overall description of the scene",
              "dense_captions": ["detailed descriptions of subregions"],
              "suggested_category": "best matching category from this list: {CATEGORIES}"
            }}

            Rules:
            - Return only JSON (no explanations, no markdown).
            - The category must be **exactly one** from the list above.
            - If no perfect match exists, choose the closest reasonable category.
            """
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and produce the structured JSON response."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    # Run GPT-5 model
    completion = client.chat.completions.create(
        model=OPENAI_model_name,
        messages=messages,
        max_completion_tokens=1000,
    )

    # Parse model output
    response_text = completion.choices[0].message.content
    try:
        gpt_result = json.loads(response_text)
    except json.JSONDecodeError:
        gpt_result = {"main_caption": None, "dense_captions": [], "suggested_category": response_text}

    # Token usage and cost estimation
    input_cost = 0.00003 * completion.usage.prompt_tokens
    output_cost = 0.00006 * completion.usage.completion_tokens
    total_price = input_cost + output_cost

    # Build the same result structure as before
    result_with_source = {
        "main_caption": gpt_result.get("main_caption"),
        "dense_captions": gpt_result.get("dense_captions", []),
        "suggested_category": gpt_result.get("suggested_category"),
        "completion_token": completion.usage.completion_tokens,
        "total_token": completion.usage.total_tokens,
        "prompt_token": completion.usage.prompt_tokens,
        "total_cost": round(total_price, 4),
    }

    return json.dumps(result_with_source, indent=2)