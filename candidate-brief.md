# AI Engineer Take-Home - Sales Assistant Agent

**Time box: about 4 hours.** Please do not spend more. We would rather see a small, honest, working agent than a large unfinished one.
**AI coding tools are allowed.** You will be asked to explain your own work, so make sure you understand everything you build.

**What we care about:** the design of your agent, how it routes each message to the right action, and how the agent actually works. This is **not** a production-grade solution and we are **not** interested in a UI. A command line is fine. Spend your time on the technical parts only.

---

## How this works

1. You design and build the assistant (details below).
2. You share your code through a public link: a public repository, or a public link to a zip file. **Do not include the `data/` folder.**
3. You record a 5–7 minute walkthrough video: an overview of the main components of your solution, and running two or three prompts live (see "What to deliver").
4. On the video interview link we sent you, you paste the two public links (code and video) and answer five questions in a video interview.
5. Some candidates are then invited to an interview.

---

## The scenario

A Metso salesperson is on a call with a mining customer. They need a chat assistant that can:

1. **Answer product questions and recommend a product** using our product brochures.
2. **Give a rough cost estimate for a field service visit** using records of past visits.
3. **(Optional) Assess a customer's flotation process readings** against historical plant data. See the optional section below.

Parts 1 and 2 are required. Part 3 is optional.

The assistant should decide for itself what to do for each message. It should ask for missing information when needed, and it should say so when it does not know something instead of guessing.

---

## What you are given (`data/`)

| Folder / file | What it is |
|---|---|
| `brochures/` | 12 product brochures. Each one is provided twice, with the same base name: the **original PDF** (`.pdf`) and a ready-made **text extraction** (`.md`, page markers included). Use whichever you prefer: work from the extracted text as-is, or extract the PDFs yourself with your own approach. The brochures are the **only** source of truth for product information. The provided text is a basic extraction, so tables and layout are flattened and some content is imperfect. |
| `HistoricalServices/` | 50 field service reports (`.txt`) written by technicians. They are the only source for cost estimates. |
| `flotation_process_data_hourly.csv` | *Optional, see below.* 4,097 hourly rows (March–September 2017) from an iron ore flotation plant: feed grade, reagent flows, pulp properties, column air flows and levels, and the quality of the concentrate (% iron, % silica). Each row is the hourly average of the plant's 20-second sensor readings. |

---

## What to build

A command-line or simple chat agent (any language, any LLM API, any agent framework) that:

- Answers product questions grounded in the brochures, with no invented specifications.

- Recommends a specific product, with a reason tied to what the customer said.
- Estimates the cost of a service visit **from the historical reports**, not invented. 
- Says clearly when the answer is not in its data.


Look at the data before you design anything. Your design should be explainable for a much larger dataset, but **you do not need to build for scale**.

### Optional (part 3): flotation process data (max 45 minutes)

A salesperson says: *"The customer is getting 3.1% silica and 64.5% iron in their concentrate. They think that's normal. What do you think, and what could we suggest?"*

Using `flotation_process_data_hourly.csv`, let the agent answer this by actually computing from the data, not from general knowledge. How far you take it is up to you; we are interested in what you find and how much you trust it. Skipping this will not count against you.

---

## What to deliver

### 1. Code (public link)

A public link to your code: either a public repository (for example on GitHub) or a public link to a zip file (for example on OneDrive or Google Drive, accessible to anyone with the link). **Do not include the `data/` folder**, add a few lines on how to run it, and do not include any API keys.

### 2. Walkthrough video (5–7 minutes)

Record your screen while you:

- give an overview of the main components of your solution;
- run these prompts live in your agent:
  1. **Product part:** "We have Larox pressure filters but performance drops whenever our feed grade changes"
  2. **Cost part:** "A customer in Chile needs 12 filter cloths replaced on a Larox PF-DS tower press. What should I budget for the service visit?"
  3. **Only if you did the optional part:** "The customer is getting 3.1% silica and 64.5% iron in their concentrate. They think that's normal. What do you think?"

If your agent asks a clarifying question, answer it with one short, realistic reply, on screen.

Upload the video somewhere that gives you a public link (for example YouTube as unlisted, OneDrive or Google Drive) and share that link.

### 3. Video interview (five questions)

We will send you a link to a video interview on the Recright platform. There you paste the public links for your code and your walkthrough video, and then record your answers to five questions.

- **Free up some time before you start.** The interview is continuous: once you begin, the questions follow one after another.
- **You will not see the questions in advance.** They are about your own solution and the choices you made while building it, so make sure you can explain every part of your work, including how you used AI tools.
- Each question gives you 1 minute to prepare and up to 5 minutes to answer. You do not have to use all the time: if you feel you have answered the question, you can move on to the next one. Up to three retakes are possible for each question.

---

## Constraints

- Core implementation must be your own code (your own code here includes also code generated with AI coding assistants, we are interested in understanding how you use them for development). No pre-built assistant platforms or no-code tools (e.g. Copilot Studio, Dust, Dify).

---

**Deadline:** the full submission, including the video interview, must be completed by **midnight on 20 October (23:59, Helsinki time)**.

If you have questions about the assignment, send an email to tarek.ali@metso.com. For technical problems with the interview platform, contact support@recright.com.
