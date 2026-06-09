# The Unofficial Guide — Project 1

A Q&A system about Georgia Tech housing. You ask where to live, and it answers
only from a set of scraped GT housing sources, listing the sources it used.

**Pipeline:** scrape → chunk (`chunk_text.py`) → embed + store (`embedding.py`,
all-MiniLM-L6-v2 + ChromaDB) → retrieve (`retrieval.py`) → generate
(`generation.py`, Groq) → Gradio UI (`app.py`).

---

## Domain

Georgia Tech housing — on-campus dorms and the off-campus neighborhoods nearby.
This is useful because the info is spread across official pages, apartment blogs,
and Reddit, and no single source captures what places actually feel like (which
dorms are quiet, which areas are cheap, East vs. West campus). Normally a student
spends hours searching, but this system does that searching once.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | GT Postdoctoral Education — nearby neighborhoods guide | Blog / institutional | https://postdocs.gatech.edu/news/prospective-campus-dwellers-guide-nearby-neighborhoods |
| 2 | r/gatech — off-campus options for grad students | Reddit | https://www.reddit.com/r/gatech/comments/1k55yq7/what_are_some_good_offcampus_housing_options_for/ |
| 3 | GT My Campus Support — Neighborhoods Around Campus | Blog / institutional | https://mycampussupport.gatech.edu/hc/en-us/articles/32070965705741-Neighborhoods-Around-Campus |
| 4 | Rambler Atlanta — Where GT students live | Blog (third-party) | https://rambleratlanta.com/resources/where-georgia-tech-students-live/ |
| 5 | RateMyDorm — GT dorms ranked by reviews | Review aggregator | https://www.ratemydorm.com/dorms-ranked/georgia-institute-of-technology |
| 6 | r/gatech — which side of campus to live on | Reddit | https://www.reddit.com/r/gatech/comments/1rompce/which_side_of_the_campus_should_i_live_in/ |
| 7 | Rambler Atlanta — freshman housing guide | Blog (third-party) | https://rambleratlanta.com/resources/freshman-guide-housing-georgia-tech/ |
| 8 | GT International Education — housing on/around campus | Blog / institutional | https://isss.oie.gatech.edu/content/housing-campus-and-around-gt |
| 9 | r/gatech — safe and decently priced off-campus | Reddit | https://www.reddit.com/r/gatech/comments/1lczd0n/safe_and_decently_priced_offcampus_housing/ |
| 10 | Platuni — off-campus housing at GT | Blog (third-party) | https://www.platuni.com/enterprise-resources/blog-and-insights/off-campus-housing-at-georgia-tech |

Scraped text lives in `GT Housing Info/`, one file per source.

---

## Chunking Strategy

Each source type is chunked differently because they're structured differently
(`chunk_text.py`).

**Chunk size:**
- **Blogs:** ~100 tokens, split on whole sentences (never mid-sentence).
- **Reddit:** one chunk per comment.
- **Reviews (RateMyDorm):** one chunk per dorm review.

**Overlap:** ~15 tokens for blogs (one trailing sentence carried over). None for
Reddit or reviews, since each comment/review is already a complete thought.

**Why this fits my documents:** A fixed token split cut blog sentences in half and
broke facts across chunks, so I switched blogs to sentence-based chunks. Reddit and
reviews are already split by comment and by dorm, so I just chunk on those.

**Preprocessing:** Before embedding I strip the citation header, Markdown (`##`,
`**`), and Reddit clutter (usernames, vote counts, reply notes). These added noise
and lowered match scores; removing them raised top-match similarity.

**Final chunk count:** 101 chunks.

**Sample chunks:
**Blog** — `07_rambler_freshman_guide.md`, 95 tokens:
> Most dorms accommodate two students per room, though triples and quads exist.
> East campus is in proximity to Greek Life, the football stadium, and Tech Square,
> while west campus offers a quieter environment and is closer to the Campus
> Recreation Center. Off-Campus Housing: These apartments provide greater autonomy
> regarding roommates, amenities, lease terms, and location within Midtown. Students
> gain access to private kitchens and can typically secure year-long leases rather
> than nine-month contracts.

**Review** — `05_ratemydorm_freshman_dorms_ranked.md`, 109 tokens:
> #5 — Freeman (based on 1 review)
> One of the newer buildings on the west campus. Connected with Montag so if you have
> friends living there, it's easy to walk downstairs to visit. There're 2 elevators
> available. Printer and laundry are in Fitten, kinda inconvenient on rainy days.
> During my freshman year, there were 3 to 4 fire alarms every semester. Some rooms
> smell funny. Freeman is moister compared to Montag. — 6 years ago

**Reddit** — `09_reddit_safe_priced_offcampus.md`, 15 tokens:
> I 1200 in home park for a studio- possible to find, just hard

**Reddit** — `02_reddit_offcampus_grad_options.md`, 2 tokens:
> Home Park

**Reddit** — `06_reddit_which_side_of_campus.md`, 156 tokens:
> I personally live on west and I like it quite a bit. Way less noise and traffic on
> this side of campus. Also from what I can tell, our parking garage seems to be much
> safer than NAV's/east. I've heard of a few people I know having their cars broken
> into over there, but I haven't seen that at all here on west. Still, I assume that's
> a relatively rare thing to happen and if you don't leave valuables in your car I'm
> sure you'll be fine. Food is pretty mid on both ends of campus. My walks to class
> are generally about 10' give or take, but I don't have any in scheller (which is
> about 30' walk).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` (sentence-transformers), stored in ChromaDB with
cosine similarity. The same model embeds the chunks and the query. I picked it
because it's small, fast, free, and good for short text like comments and reviews.

**Production tradeoff reflection:** With no cost limit, the main tradeoff is accuracy
vs. speed/size. MiniLM is small and only gives moderate scores (0.4–0.7) and caps at
256 tokens. A bigger model like `all-mpnet-base-v2`, or one built for question→passage
search, would rank the right chunk higher — but it's slower and heavier. My chunks are
short and English-only, so accuracy on housing text would be the deciding factor.

Sample retrieval tests:
1. Where should I live off campus if I want to be near MARTA and I am a runner

[5] sim=0.504  Housing on Campus and Around GT
      ISSS provides a hotel listing with price guides ($ to $$$$) and transit directions to campus. The nearest MARTA stations are North Avenue and Midtown; the Tech 

Marta is mentioned here, so I do belive this is a successful pull.

2. Where should I live on campus as a freshmen if I enjoy quiet?
[1] sim=0.652  Reddit — r/gatech: Which side of the campus should I live in?
      West campus has always been my go-to recommendation. Want to go party on east campus? It's a 15-20 minute walk away. Want to study or sleep in peace? You're 15 

Both sides of campus are mentioned here, so I do believe this is relevant pull.

3. What is east campus like?
  [1] sim=0.742  Reddit — r/gatech: Which side of the campus should I live in?
      East campus is home to Grand Challenges and Global Leadership LLCs. Going out, partying, football tailgating, and sorority/fraternity culture is bigger there si

---

## Grounded Generation

Grounding is enforced in code, not just asked for in the prompt (`generation.py`).

**System prompt:** The model is told to answer only from the given passages, with the
line *"Do not use any outside or prior knowledge,"* and to say *"The provided sources
don't cover that"* when it can't answer. It runs at `temperature=0`.

Two things back this up:
1. **Retrieval gate:** if the top match scores below 0.30, the code returns a fixed
   "I didn't find much" message and never calls the LLM — so it can't make something up.
2. **Sources are built in code** from the retrieved chunks' metadata, not from the
   model's text, so the source list is always correct even if the model forgets to cite.

**How sources are shown:** every answer ends with a numbered Sources list (clickable
links in the UI), built from the chunks that were actually used.

---

## Evaluation Report

| # | Question | Expected | System response | Retrieval | Accuracy |
|---|----------|----------|-----------------|-----------|----------|
| 1 | Quieter place to live on campus? | West Campus | Says West Campus — quieter, less noise. | Relevant | Accurate |
| 2 | Off-campus near MARTA if I'm a runner? | Midtown | The neighborhood near a MARTA station that may be suitable for a runner is Midtown, as it provides Piedmont Park and its abundance of trails | Partial relevant | Partially accurate |
| 3 | Cheapest off-campus option? | Home Park | Says "$750 at 464 Calhoun St" (one Reddit listing). | Partially relevant | Partially accurate |
| 4 | On-campus options after freshman year if I join a sorority? | Greek houses |Greek housing represents another on-campus option, with 24% of students involved with Greek life at Georgia Tech, though only a fraction can live in fraternity or sorority houses | Relevant | Accurate |
| 5 | What do people say about East campus? | Social and lively | People say that East campus is home to a lively atmosphere, with a bigger partying, football tailgating, and sorority/fraternity culture, and is closer to Midtown, the football stadium, and Tech Square. However, it's also said to be noisier and louder at night. East campus is considered good for those who want to be as social as possible and engross themselves in campus activities [4]. | Relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

Result: 3 accurate, 2 partially accurate. 

Refusal:
Question - What should I have for lunch?
Answer -  ⚠️ Low confidence — the sources may not cover this well.

I didn't find much in my sources about that. The guide covers Georgia Tech on-campus dorms and nearby off-campus neighborhoods, so try rephrasing or asking about one of those.

---

## Failure Case Analysis

**Question that failed:** Off-campus near MARTA if I'm a runner?

**What the system returned:** You can consider living in Downtown Atlanta, which offers competitive rent and MARTA access. Alternatively, you can look into Atlantic Station, which is west of campus and has shuttle services, or West Midtown, which has contemporary lofts and is near Atlantic Station. These areas may also be suitable for running.

**Root cause:** This is a retrieval problem. The chunks retrieved for this question look like   k=5  |  low_confidence=False
  [1] sim=0.592  Reddit — r/gatech: Which side of the campus should I live in?
      I personally live on west and I like it quite a bit. Way less noise and traffic on this side of campus. Also from what I can tell, our parking garage seems to b
  [2] sim=0.579  Off-Campus Housing at Georgia Tech | What to Know
      2. Home Park — North of campus, popular for affordable single-family homes and duplexes within walking distance. 3. Atlantic Station — West of campus with moder
  [3] sim=0.540  Reddit — r/gatech: Which side of the campus should I live in?
      East: Closer to football Stadium, frats, and Midtown; usually busier. West: Closer to Campus Rec Center (CRC); usually quieter; Music Building is on West. Only 
  [4] sim=0.529  Reddit — r/gatech: Safe and decently priced off-campus housing
      Like one of the comments here, I recommend West Midtown/near Atlantic Station for safety, more reasonable prices, and proximity to campus. If you get a roommate
  [5] sim=0.504  Housing on Campus and Around GT
      ISSS provides a hotel listing with price guides ($ to $$$$) and transit directions to campus. The nearest MARTA stations are North Avenue and Midtown; the Tech 
Where none of them return anything about Midtwon and Piedmont park.
**What I would change:** Phrasing my question in a different way can sometimes yield different results.

---

## Spec Reflection

**How the spec helped:** Writing the Chunking Strategy in planning.md first made me
decide that blogs and Reddit needed different handling. That became the structure of
`chunk_text.py` directly, and the "4–8 chunks" note turned into my `suggest_k()`
function.

**How my build diverged:** I planned 500-token blog chunks but ended up at ~100-token
sentence-based chunks. The 500-token ones split facts across chunks, and the back half
was past MiniLM's 256-token limit so it wasn't even embedded. I also added a per-review
chunk mode for RateMyDorm and used Groq's `llama-3.3-70b` instead of the GPT-4o in my
diagram, since the project is set up for a free Groq key.

---

## AI Usage

**Instance 1 — Scraping the sources**
- *What I asked:* Scrape my 10 URLs into one file each — keep blog text + citation,
  Reddit question + all comments, and dorm details.
- *What it produced:* A cleaned file per source in `GT Housing Info/`.
- *What I changed:* Three Reddit threads and one GT page were blocked, so I pasted that
  text in by hand. I also swapped source 5 (the GT housing page had no scrapable dorm
  info) for RateMyDorm's review rankings.

**Instance 2 — Building `chunk_text()`**
- *What I asked:* Implement my blog-vs-Reddit chunking with my size and overlap.
- *What it produced:* A 500-token sliding window plus per-comment Reddit splitting.
- *What I changed:* I cut blog size to 100, switched blogs to sentence-based chunks
  after seeing mid-sentence cuts, and added a cleaning step to strip Markdown and Reddit
  noise.

**Instance 3 — Building the app**
- *What I asked:* Answers from the sources only, with guaranteed source citations.
- *What it produced:* A `generate_answer()` plus the Gradio interface but with the sources cited directly after each statement.
- *What I changed:* I did not want each individual sentence to be cited. Only the sources to be cited below of what was used to generate the answer.
