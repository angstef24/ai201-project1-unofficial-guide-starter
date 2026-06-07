# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
I chose my domain of Georgia Tech housing options. There are so many websites out there that give a lot of information, but it is hard to gather it all without spending a large amount of time on your computer. So, instead I'll be spending the time on my computer so a future student can spend less time on theirs.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 |Gatech Website|Guide to Off Campus Housing|https://postdocs.gatech.edu/news/prospective-campus-dwellers-guide-nearby-neighborhoods|
| 2 |Reddit|Discussion on Off Campus Housing|https://www.reddit.com/r/gatech/comments/1k55yq7/what_are_some_good_offcampus_housing_options_for/|
| 3 |My Campus Passport|Neighborhoods Around Gatech|https://mycampussupport.gatech.edu/hc/en-us/articles/32070965705741-Neighborhoods-Around-Campus|
| 4 |Rambler|Where Students Live|https://rambleratlanta.com/resources/where-georgia-tech-students-live/|
| 5 |RateMyDorm|GT freshman dorms ranked by student reviews|https://www.ratemydorm.com/dorms-ranked/georgia-institute-of-technology|
| 6 |Reddit|Where should I live on campus|https://www.reddit.com/r/gatech/comments/1rompce/which_side_of_the_campus_should_i_live_in/|
| 7 |Rambler|Freshmen guide to housing|https://rambleratlanta.com/resources/freshman-guide-housing-georgia-tech/|
| 8 |International Education Website|On and Off campus housing blog|https://isss.oie.gatech.edu/content/housing-campus-and-around-gt|
| 9 |Reddit|Off campus discussion|https://www.reddit.com/r/gatech/comments/1lczd0n/safe_and_decently_priced_offcampus_housing/|
| 10 |Platuni|Third party blog on off campus information|https://www.platuni.com/enterprise-resources/blog-and-insights/off-campus-housing-at-georgia-tech|

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

Chunk size: For long blog websites, lets chunk by 500 tokens in order to split up the content. For reddit pages, I want each chunk to be each comment left on the page.

Overlap: No overlap needed for reddit posts, but for blog posts lets add a 52 token overlap.

Reasoning: Overlap matters more for the blog post websitw where we are splitting based on a fixed number of tokens. On reddit, each comment is its own complete thought anyways, so adding an overlap will not serve a purpose. However, it is important to split these types up since the same chunking strategy would not be successful for both.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

Embedding model: all-MiniLM-L6-v2 via sentence-transformers

Top-k: Let's retreive around 4-8 retreieved chunks. We may need less chunks if the question is factual. However, we should strive to retrieve closer to 8 when the question is opinionated to provide the most conetxt.

Production tradeoff reflection:
The higher the k value means there is the potential for less relevent chunks to get added in if there is not that much information on the question being asked.
If the top retrieved chunks all have low similarity scores, we should surface that to the user — "I didn't find much on this, here's what I have" — rather than generating a confident answer from weak evidence.
---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 |Where should I live on GT's campus if I am looking for a quieter place to live?|West Campus|
| 2 |What neighborhood should I look to live in off campus if I want to be near a MARTA station, and I am a runner?|Midtown|
| 3 |Where is my cheapest option for living off campus?|Home Park|
| 4 |What are my options for living on campus after freshman year if I join a soroity?|Greek houses|
| 5 |What do people say about living on East campus?|It is much more social and lively.|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. I think that the websites could hold inconsistent data especially with price and availability since GT is constantly changing the housing situation. Older websites may prove less useful and cloud the model with irrelevant chunks.

2. There could be not enough opinionated data, since most websites have just information allowing the user to determine what their opinion of this information is.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->
+---------------------------+
|    Document Ingestion     |
|  BeautifulSoup / PRAW     |
|  GT blogs + Reddit        |
+---------------------------+
             |
             v
+---------------------------+
|         Chunking          |
|  LangChain                |
|  500 tokens, 50 overlap   |
|  Full comments            |
+---------------------------+
             |
             v
+---------------------------+
|  Embedding + Vector Store |
|  all-MiniLM-L6-v2         |
|  ChromaDB                 |
+---------------------------+
             |
             v
+---------------------------+
|         Retrieval         |
|  ChromaDB cosine search   |
|  k=5-8 chunks             |
+---------------------------+
             |
             v
+---------------------------+
|        Generation         |
|  GPT-4o                   |
|  returns cited answer     |
+---------------------------+
---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

    

**Milestone 3 — Ingestion and chunking:**
I will be using Claude as my AI tool. I will give Claude first the domain to understand the context behind what is being asked. Next, I will have Claude build code that will scrap all of my files and save the information each in its own file in a folder called GT Housing Info. Then I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
with my specified chunk size and overlap using the files it just generated. 
**Milestone 4 — Embedding and retrieval:**
 After verifying the chunking strategy is providing useful information, I will have Claude set up the retrieval code by first setting up the embedding model and then the retrieval function that returns my top-k amount of 4-8 chunks. 
**Milestone 5 — Generation and interface:**
Finally we will build the generation and interface code. I will make sure my model with only pull information from my files.