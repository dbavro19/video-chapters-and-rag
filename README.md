
# Video Chapters and RAG

This is sample code demonstrating the use of Generative AI's capability to analyze and identify topics/sections in unmarked video content and provide a searchable interface that uses Generative AI and Retrieval Augmeented Generation to respond to end user questions with video content and links to the start time of relevant video. 

![Architecture and workflow diagram](https://github.com/dbavro19/video-chapters-and-rag/blob/main/video-sections-Fuzzy-v2-RAG.drawio.png)

This project is a prototype/proof of concept and should not be leveraged directly in production

It should also be noted that this project focuses soley on the audio content within a video, and was primarily designed with Education video content in mind

## Features

- Uploads a video file to S3
- transcribe the audio content into text
- Generate section titles, summaries, and start times using multiple Generative AI prompts and models
- Creates vector embeddings of the generated video summaries 
- Store the vectorized data in an OpenSearch datastore
- Search and retrieve relevant video sections based on user queries using KNN and semantic search
- Respond with video clips at the relevant start time

## Two Workflows
 - ingest-app.py focuses on the uploading, transcribing, and segment/start time identification of the uploaded video. And the embedding and persisting of that content into a vector store
- search-app.py provides a natural language interface to search for relevant video content that returns a summary of the video along with a link to the video itself at the relevant start time.

## Prerequisites

- AWS account with access to Anthropic's Claude Instant and Claude 2.1 model via Amazon Bedrock
- OpenSearch Serverless Vector store 
- CloudFront (or a public S3 bucket) for serving the video content
- Python 3.x


## Installation

1. Clone the repository:

```bash
git clone https://github.com/dbavro19/video-chapters-and-rag.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the necessary AWS services (Bedrock, Transcribe, S3, OpenSearch, CloudFront) and configure the appropriate credentials.

4. Update the code ingest-app.py and search-app.py files to apply your S3 bucket name, Opensearch Host and Index Name, and CloudFront distribution

## Usage

1. Run the Streamlit appa:

```bash
streamlit run ingest-app.py
```

```bash
streamlit run search-app.py
```

2. In ingest-app.py: Upload a video file through the Streamlit interface.
3. The app will automatically transcribe the audio, generate section titles, summaries, and start times using Generative AI.
4. If happy with the results, select "Save" and the generated content will be vectorized and stored in the OpenSearch datastore.
5. In search-app.py: You can search for relevant video sections by entering queries in the provided search interface.
6. The app will respond with relevant video clips and timestamps using Retrieval Augmented Generation (RAG).


## NOTE

The contents of this repository represent my viewpoints and not of my past or current employers, including Amazon Web Services (AWS). All third-party libraries, modules, plugins, and SDKs are the property of their respective owners.

```
![image](https://github.com/dbavro19/video-chapters-and-rag/assets/12836698/937c6584-ec4a-4d71-a68b-a8e1378f444c)
