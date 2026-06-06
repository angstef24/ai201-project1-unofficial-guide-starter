# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

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
| 5 |Gatech Website|On campus housing options|https://housing.gatech.edu/explore-housing/on-campus-housing|
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

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

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

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
