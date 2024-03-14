import boto3
import time
import requests
import json
import botocore
import streamlit as st
import time
from datetime import datetime
from langchain.text_splitter import CharacterTextSplitter #using for text splitter only
from thefuzz import fuzz
import pandas as pd
from opensearchpy import OpenSearch
from opensearchpy import RequestsHttpConnection, OpenSearch, AWSV4SignerAuth


config = botocore.config.Config(connect_timeout=300, read_timeout=300)
bedrock = boto3.client('bedrock-runtime' , 'us-east-1', config = config)

#Streamlit setup
### Title displayed on the Streamlit Web App
st.set_page_config(page_title="Create Video Sections and Timestamps", page_icon=":tada", layout="wide")


#Header and Subheader dsiplayed in the Web App
with st.container():
    st.header("Upload a video and automatically create relevant section and timestamps based on the audio from the video using GenAI")
    st.subheader("")
    st.title("")

#upload video to s3 (return object_key name)
def upload_to_s3(file, file_name):
    bucket="BUCKET_NAME" #Add your bucket name here (prototyping purposes only)
    object_key = file_name.strip()

    s3 =boto3.client('s3')
    response = s3.upload_fileobj(file, bucket, object_key)
    print(response)

    return object_key

#returns the hardcoded CloudFront name if you want to serve video's directly via CloudFront - otherwise you can built the video's s3 uri and use an S3 bucket directly
def get_cloudfront_name(object_name):
    cf_url=f"CLOUDFRONT_DISTRO_NAME{object_name}" #Add your CloudFront Origin here
    return cf_url


#transcription job - returns full transcript and subtitled (aka timestamped) transcript
def transcribe_file(object_name): 

    transcribe_client = boto3.client('transcribe')

    file_uri= f"s3://BUCKET_NAME/{object_name}"
    job_name=object_name+time.strftime("%Y%m%d-%H%M%S")
    full_transcript=""

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': file_uri},
        #MediaFormat='wav',
        LanguageCode='en-US',
                Subtitles = {
            'Formats': [
                'srt'
            ],
            'OutputStartIndex': 0 
       }
    )

    max_tries = 60
    while max_tries > 0:
        max_tries -= 1
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = job['TranscriptionJob']['TranscriptionJobStatus']
        if job_status in ['COMPLETED', 'FAILED']:
            print(f"Job {job_name} is {job_status}.")
            if job_status == 'COMPLETED':
                print(
                    f"Download the transcript from\n"
                    f"\t{job['TranscriptionJob']['Transcript']['TranscriptFileUri']}.")
                
                job_result = requests.get(
                    job['TranscriptionJob']['Transcript']['TranscriptFileUri']).json()
                full_transcript=job_result['results']['transcripts'][0]['transcript']

                #testing
                print("______________________")
                print(job)

                sub_url=job['TranscriptionJob']['Subtitles']['SubtitleFileUris'][0]

                print(sub_url)

                #sub_url=job['Subtitles']['SubtitleFileUris'][0]

                transcript_response = requests.get(sub_url)
                full_subtitles=transcript_response.content.decode()

                print("----Full Transcript ----")
                print(full_transcript)
                print("----Full Transcript ----")

                print("----Full Subtitles ----")
                print(full_subtitles)
                print("----End Subtitles ----")

                return full_transcript, full_subtitles
            break
        else:
            print(f"Waiting for {job_name}. Current status is {job_status}.")
        time.sleep(10)


#Invoke Bedrock LLM - Returns JSON Array of Topics, Summaries, and Starting Sentences 
def Create_topics(transcription, title):


     ##Setup Prompt
    prompt_data = f"""
Human:
You will be provided a video transcript
Using the transcript, identify and respond with all the main sections of the video. This should include introduction and conclusion sections, if relevant sections are found in the transcription
The Sections should be in order and capture all the contents of the video
For every section identified, create a short Section Title and a detailed Section summary that summarizes the ENTIRE context of the section
Provide the first sentence from the video_transcription that begins the section. This sentence should mark the start of the section you have identified and should mark the transition into the section
Return the Section Titles,  Summaries, and beginning sentence in a valid JSON array

Video Title:
<title>
{title}
</title>

Here is the provided video transcript with timestamps
<video_transcription>
{transcription}
</video_transcription>

Use the following format as a guide for your output
<output_format>
{{
{{"Title": (Short Section Title), "Summary":  (Summary of Topic), "Starting_Sentence":  (Sentence that starts the Section)}},
{{"Title": (Short Section Title), "Summary":  (Summary of Topic), "Starting_Sentence":  (Sentence that starts the Section)}}
}}
<output_format>

Please return the JSON formatted response for each identified section response in <response> XML Tags without any new lines

Assistant:
"""


    body = json.dumps({"prompt": prompt_data,
                 "max_tokens_to_sample":10000,
                 "temperature":0,
                 "top_k":250,
                 "top_p":0.5,
                 "stop_sequences":[]
                  }) 
    
    #Run Inference
    modelId = "anthropic.claude-v2:1"  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body.get('completion')

    result=parse_xml(llmOutput, "response")

    return result

#Manipulate context
def split_transcript(subtitles):


    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500, #Testing with hard coded 500
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    parts = text_splitter.create_documents([subtitles])

    return parts


#Get Starting TimeStamps
def starting_time(subtitles, sentence):

    #Get starting time for all sections
    #Depending on use case, you may want to hardcode the first section to 00:00:00

      ##Setup Prompt
    prompt_data = f"""
Human:
Return the earliest "Start timestamp" associated with the <focus_sentence> from the provided <subtitles>
The associated sentence you return the timestamps from should be the closest possible match in the provided <focus_sentence>
Timestamp output should be in (hh:mm:ss,msms) format

<focus_sentence>
"{sentence}"
<\focus_sentence>

Use the format_info as a guide to interpret the subtitles format:
<format_info>
(Section number - indicated the Section number or paragraph number - dont return this as a timestamp value)
("Start timestamp" of sentence in hh:mm:ss,msms format)  --> ("End timestamp" stamp in hh:mm:ss,msms format)
(Sentence associated with the above timestamp)
<\format_info>

<subtitles>
{subtitles}
<\subtitles>

Return the start and end timestamps you found for the associated sentence in <timestamps> xml 
Return the earliest Start timestamp for the associated sentence in hh:mm:ss,msms format in <start_time> xml tags

Assistant:
"""
    body = json.dumps({"prompt": prompt_data,
                 "max_tokens_to_sample":1000,
                 "temperature":0,
                 "top_k":100,
                 "stop_sequences":[]
                  }) 
    
    #Run Inference
    #modelId = "anthropic.claude-v2:1"  # change this to use a different version from the model provider if you want to switch 
    modelId = "anthropic.claude-instant-v1"  # Results with Claude instant have been reliable and much faster
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body.get('completion')

    print("------PRINTING LLM OUTPUT----------")
    print(llmOutput)
    print("------DONE--------")

    start_time=parse_xml(llmOutput, "start_time")
    #end=parse_xml(llmOutput, "end")
    #thoughts = parse_xml(llmOutput, "thoughts")

    return start_time


#parse xml
def parse_xml(xml, tag):
    temp=xml.split(">")
    
    tag_to_extract="</"+tag

    for line in temp:
        if tag_to_extract in line:
            parsed_value=line.replace(tag_to_extract, "")
            return parsed_value


#Calculate seconds for timestamps in hh:mm:ss,msms format
def time_math_seconds(timecode):
    print(timecode)
    times=timecode.split(":")
    hour=int(times[0])*60
    minute=int(times[1])*60
    seconds_temp=times[2].split(",")
    seconds = int(seconds_temp[0])

    
    suffix = hour+minute+seconds
    return suffix

def fuzzy_search(topic_sentence, parts, num_sections, total_sections): #basically removing the fist and last section of the split docuemnt unless its the first or last topic. Intro's and conclusions (particualrly names) are throwing off the fuzzy score

    segments = parts.copy()
    
    partial_ratio_score=0
    partial_ratio_item = ""


    if total_sections < 3: #handle small video's by not removing any sections
        for item in segments:
            x = fuzz.partial_ratio(topic_sentence, item.page_content)
            print(x)
            if x > partial_ratio_score:
                partial_ratio_score = x
                partial_ratio_item = item.page_content
                print("new x! with score of " + str(partial_ratio_score))

        print(partial_ratio_item)
        return(partial_ratio_item)

    elif num_sections < 3: # if its the first 2 sections section, remove the last part of index (intros semantically are very similar to conclusions in educational videos)
        last = (len(parts)) -1
        segments.pop(last)
        print(len(segments))

        for item in segments:
            x = fuzz.partial_ratio(topic_sentence, item.page_content)
            print(x)
            if x > partial_ratio_score:
                partial_ratio_score = x
                partial_ratio_item = item.page_content
                print("new x! with score of " + str(partial_ratio_score))

        print(partial_ratio_item)
        return(partial_ratio_item)

    elif num_sections == total_sections: # if its the last section, remove the first part of index
        first=0
        segments.pop(first)

        for item in segments:
            x = fuzz.partial_ratio(topic_sentence, item.page_content)
            print(x)
            if x >= partial_ratio_score:
                partial_ratio_score = x
                partial_ratio_item = item.page_content
                print("new x! with score of " + str(partial_ratio_score))
                
        print(partial_ratio_item)
        return(partial_ratio_item)        

    else: # Anywhere else, remove the first and last parts 
        first=0
        last = (len(parts)) -1
        segments.pop(last)
        segments.pop(first)

        for item in segments:
            x = fuzz.partial_ratio(topic_sentence, item.page_content)
            print(x)
            if x >= partial_ratio_score:
                partial_ratio_score = x
                partial_ratio_item = item.page_content
                print("new x! with score of " + str(partial_ratio_score))
                
        print(partial_ratio_item)
        return(partial_ratio_item)        



#method for adding new row into final dataframe output   
def add_row(df, new_row_data):
    # Creating a DataFrame from the new row data
    new_row_df = pd.DataFrame([new_row_data], columns=df.columns)
    # Concatenating the existing DataFrame with the new row DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)
    return df


#Persisting docuemnts into Opensearch
def persist_doc(doc):
    #Setup Opensearch connectionand clinet
    host = '14dzfsbbbt70yuz57f23.us-west-2.aoss.amazonaws.com' #use Opensearch Serverless host here
    region = 'us-west-2'# set region of you Opensearch severless collection
    service = 'aoss'
    credentials = boto3.Session().get_credentials() #Use enviroment credentials
    auth = AWSV4SignerAuth(credentials, region, service) 

    oss_client = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        pool_maxsize = 20
    )

    tempdf = pd.DataFrame(doc)

    for row in tempdf.iterrows():
        #title = row['Title']
        #summary = row['Summary']
        #start_time = row['Start Time']
        #video_link = row['Video Link']

        st.write(row[1])

        title = row[1]['Title']
        summary = row[1]['Summary']

        start_time_temp = row[1]['Start Time']
        start_time = time_math_seconds(start_time_temp.strip())

        video_source = row[1]['Video Link']

        #Get Embeddings - returns vectorized value of input string
        vectors = get_embeddings(bedrock, summary)
        #Index document
        response = index_doc(oss_client, vectors, title, summary, video_source, start_time)

        print(response)

        st.write("saved")

    return "Done"
        




#Index document
def index_doc(client, vectors, title,summary,video_source, source_seconds):


        
    indexDocument={
        'vectors': vectors,
        'Title': title,
        'Summary': summary,
        'StartTimeSeconds' : source_seconds,
        'VideoSource': video_source
        }

    response = client.index(
        index = "video-summaries", #Use your index 
        body = indexDocument,
    #    id = '1', commenting out for now
        refresh = False
    )
    return response


#Get Embeddings - returns vectorized value of input string
def get_embeddings(bedrock, text):
    body_text = json.dumps({"inputText": text})
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType='application/json'

    response = bedrock.invoke_model(body=body_text, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    return embedding





#setup dataframe
data = {
    'Title': [],
    'Summary': [],
    'Start Time': [],
    'Video Link': []
}

df=pd.DataFrame(data)

#Streamlit Logic

#Upload Video File
uploaded_file = st.file_uploader("Choose a Video")

result=st.button("Upload Video and Start")
if result:
    with st.status("Processing Request", expanded=False, state="running") as status:
        

        upload_start = datetime.now()
        filename= uploaded_file.name

        #upload file to s3
        status.update(label="Uploading File to S3: ", state="running", expanded=False)
        object_name=upload_to_s3(uploaded_file, filename)

        cf_name = get_cloudfront_name(object_name) #To avoid setting my AWS S3 Bucket to public, i want to serve my data via Cloudfront - this will get the Object's URI from CLoudfront

        upload_end = datetime.now()
        upload_time= upload_end - upload_start

        
        st.write(":heavy_check_mark: Uploaded File to S3: " + str(upload_time))

        #transcribe audio
        status.update(label="Transcribing Video: ", state="running", expanded=False)
        transcribe_start = datetime.now()

        transcripts = transcribe_file(object_name)

        transcribe_end = datetime.now()
        transcribe_time = transcribe_end - transcribe_start

        st.write(":heavy_check_mark: Video Transcribed: " + str(transcribe_time))


        full_transcript=transcripts[0]
        subtitles=transcripts[1]


        #Split up subtitles into x number of chars by line to use for fuzzy search
        subtitle_doc = split_transcript(subtitles)

        #create topics
        status.update(label="Identifying Video Topics: ", state="running", expanded=False)

        topic_start = datetime.now()

        topics=Create_topics(full_transcript, object_name)
        json_topics = json.loads(topics)


        topic_end = datetime.now()
        topic_time = topic_end - topic_start

        st.write(":heavy_check_mark: Topics Identified: ): " + str(transcribe_time))

        #loops through the Json and displays Title and summary
        total_sections = len(json_topics)

        start_time_start = datetime.now()

        status.update(label="Finding Start Times: ", state="running", expanded=False)


        num_sections = 1
        for key in json_topics:
            title = key['Title']
            description = key['Summary']
            topic_sentence = key['Starting_Sentence']



        #Fuzzy Partial Ratio Score as Search mechanism
            fuzzy_results = fuzzy_search(topic_sentence, subtitle_doc, num_sections, total_sections)
            start_time_fuzzy = starting_time(fuzzy_results, topic_sentence)

            #yt_suffix_fuzzy = time_math(start_time_fuzzy)
            video_time = time_math_seconds(start_time_fuzzy.strip())

            #write data into dataframe
            new_row_data = {'Title': title, 'Summary': description, 'Start Time': start_time_fuzzy, 'Video Link': cf_name}
            df = add_row(df, new_row_data)


            #play video at timestamp

            st.write(title+ ": ")
            st.video(cf_name, format="video/mp4", start_time=video_time)

            num_sections += 1

            #End of Loop

        start_time_end = datetime.now()
        start_time_full = start_time_end - start_time_start

        st.write(":heavy_check_mark: Start Times Found :"  + str(start_time_full))

        final_time = start_time_end - upload_start
        #complete status
        status.update(label=":heavy_check_mark: Request Complete: Total Time: " + str(final_time) , state="complete", expanded=False)


    st.write(df)
    if 'df' not in st.session_state:
        st.session_state.df = df

#Persist to VectorStore (OpenSearch in my case)
save_results=st.button("Save")
if save_results:
    if 'df' not in st.session_state:
        st.write("No Data To Save")
    else:
        st.write("saving...")
        persist_doc(st.session_state.df)

