import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from openai import OpenAI
from PIL import Image
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the OpenAI client
client = OpenAI(api_key="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")

# Initialize LangChain ChatOpenAI
chat_llm = ChatOpenAI(api_key="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", model="gpt-4")

def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def display_images(image_paths):
    plt.figure(figsize=(15, 10))
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

def extract_items_from_prompt(prompt):
    # Define a LangChain prompt template
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="""
        User Prompt: {prompt}

        Task: Extract the types of furniture mentioned in the user prompt. Return the results as a comma-separated list.
        """
    )

    # Create a LangChain LLMChain
    chain = LLMChain(llm=chat_llm, prompt=prompt_template)

    # Run the chain
    response = chain.run(prompt=prompt)

    # Extract the response and split into a list of items
    items = response.strip().split(",")
    items = [item.strip().lower() for item in items]  # Clean up the items

    return items

def evaluate_options(item, options, user_prompt):
    # Define a LangChain prompt template for evaluation
    prompt_template = PromptTemplate(
        input_variables=["item", "options", "user_prompt"],
        template="""
        Context:
        {options}

        Question: {user_prompt}

        Task: Evaluate the options above and choose the best {item} for the user's query. Provide a detailed explanation for your choice.
        """
    )

    # Create a LangChain LLMChain
    chain = LLMChain(llm=chat_llm, prompt=prompt_template)

    # Run the chain
    response = chain.run(item=item, options=options, user_prompt=user_prompt)

    return response

# Path to the CSV file
csv_file_path = r"C:\Users\owner\AR_RAG\test.csv"

# Read the CSV file
df = read_csv(csv_file_path)
if df is None:
    exit()

# Example user prompt
user_prompt = "bed and table suitable for room of coffee color"

# Step 1: Extract items from the user prompt
items = extract_items_from_prompt(user_prompt)
print(f"Extracted Items: {items}")

# Step 2: Generate embeddings for prompts in the CSV
prompts = df["prompt"].tolist()
prompt_embeddings = generate_embeddings(prompts)

# Step 3: Generate embedding for the user prompt
user_prompt_embedding = generate_embeddings([user_prompt])[0]  # Pass as a list and take the first result

# Step 4: Compute cosine similarity between the user prompt embedding and prompt embeddings
similarities = []
for prompt_embedding in prompt_embeddings:
    similarity = cosine_similarity(user_prompt_embedding, prompt_embedding)
    similarities.append(similarity)
print(f"Length of similarities: {len(similarities)}")
print(f"Contents of similarities: {similarities}")

# Step 5: Find the top N most relevant prompts for each item
top_n = 2  # Number of top prompts to consider per item
results = {}

for item in items:
    # Filter prompts that contain the item
    item_prompts = [prompt for prompt in prompts if item in prompt.lower()]
    item_indices = [i for i, prompt in enumerate(prompts) if prompt in item_prompts]
    
    # Get similarities for the filtered prompts
    item_similarities = [similarities[i] for i in item_indices]
    
    # Get top N indices for the item
    top_indices = np.argsort(item_similarities)[-top_n:][::-1]
    top_indices = [item_indices[i] for i in top_indices]  # Map back to original indices
    
    # Store results
    results[item] = {
        "prompts": [prompts[i] for i in top_indices],
        "image_paths": [df.iloc[i]["image_path"] for i in top_indices]
    }

# Step 6: Use LangChain to evaluate and choose the best solution for each item
final_responses = {}
for item, data in results.items():
    context = "\n\n".join([f"Option {i+1}: {prompt}" for i, prompt in enumerate(data["prompts"])])
    final_response = evaluate_options(item, context, user_prompt)
    final_responses[item] = final_response
    print(f"\nFinal Response for {item}: {final_response}")

# Step 7: Display the images corresponding to the best solutions
for item, data in results.items():
    chosen_option = None
    final_response = final_responses[item]
    
    # Extract the chosen option from the LLM's response (e.g., "Option 1")
    for i in range(top_n):
        if f"Option {i+1}" in final_response:
            chosen_option = i
            break

    if chosen_option is not None:
        print(f"\nChosen Option for {item}: Option {chosen_option + 1}")
        print(f"Image Path: {data['image_paths'][chosen_option]}")
        display_images([data['image_paths'][chosen_option]])
    else:
        print(f"No specific option was chosen by the LLM for {item}.")