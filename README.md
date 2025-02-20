<a id="readme-top"></a>
## Demo Video

<div align="center">
  <iframe width="560" height="315" 
    src="https://www.youtube.com/embed/1jX8XyYvtWs?si=XwBFWwnloh6JJzjU" 
    title="Demo Video" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

<div align="center">

</div>

<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/MrSa3dola/AR_RAG/blob/main/README.md">
    <img src="images/Ablakash.svg" alt="Logo" width="300" height="300">
  </a>
  <p align="center">
    <!--    <a href="https://drive.google.com/file/d/1uHvKp35EkJaHpnmUyHCeTQzPx6RouVmV/view?usp=drive_link">View Demo</a>-->
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
<div align="center">
  <img src="images/Screenshot.jpg" alt="Screenshot" width="320" height="620">
</div>

This project is a chatbot for furniture recommendation using multi-agents. It integrates multiple AI technologies to provide an intelligent and interactive experience for users.

### Key Features:
- Scraped IKEA data for building a product database.
- Used Florence-2-large for image captioning.
- Converted images to 3D models using TRELLIS for AR visualization.
- Stored & searched embeddings in Pinecone using mpnet-base-v2.
- Used CrewAI with three agents (RAG, Web Scraper, and Chat) powered by Gemini LLM to handle queries intelligently.
- Built the backend with FastAPI (deployed on Azure) and developed the mobile app using Kotlin.
- Under development: Image recommendation system using Vision RAG, leveraging CLIP for embedding, FAISS for similarity search, and BLIP for semantic ranking.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With
- FastAPI
- Pinecone
- CrewAI
- Florence-2-large
- TRELLIS
- Kotlin
- FAISS
- BLIP

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- FastAPI
- Uvicorn
- Pinecone client
- Kotlin (for mobile app development)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/MrSa3dola/AR_RAG.git
   cd AR_RAG
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the FastAPI backend:
   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap
- [x] Scrape IKEA data
- [x] Implement image captioning with Florence-2-large
- [x] Convert images to 3D models with TRELLIS
- [x] Integrate embedding search using Pinecone
- [x] Develop AI agents with CrewAI
- [x] Deploy backend on Azure
- [x] Develop Kotlin mobile app
- [ ] Implement Vision RAG for image recommendations
- [ ] Improve chatbot responses with fine-tuned Gemini LLM
- [ ] Add multilingual support
  - [ ] Arabic
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact
Saad Mohamed - [GitHub](https://github.com/MrSa3dola) - saad.mohamed@example.com

Project Link: [https://github.com/MrSa3dola/AR_RAG](https://github.com/MrSa3dola/AR_RAG)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
