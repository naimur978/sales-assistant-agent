# Sales Assistant Agent

A command-line sales assistant agent for a mining sales call. It answers product questions from brochures, estimates field service costs from historical reports, and assesses flotation process readings.

## Agenda

- **Agenda 1, fix the project agenda:** I write down what I will build, what I will skip, my tools, and my time plan, so I do not drift later.
- **Agenda 2, chat with Gemini:** I check that my key and setup work by chatting in the terminal.
- **Agenda 3, read the data :** I look at the brochures, reports, and the CSV rows by hand to understand what they look like.
- **Agenda 4, brochure search (Question 1, Larox):** I split the brochures by page, keep the file name and page number, and build a search that returns a score.
- **Agenda 5, cost estimate (Question 2, Chile):** I turn the 50 reports into a table once, then write plain code that gives a cost range from similar visits.
- **Agenda 6, flotation tool (Question 3, silica and iron):** I compute the answer from the CSV with `pandas`, like how 3.1% silica and 64.5% iron compare with the historical hours.
- **Agenda 7, agent loop (Questions 1, 2 and 3):** I connect all three tools to the chat with clear tool descriptions, input checks with Pydantic, a max step limit, temperature 0, and slot filling so it asks for missing details.
- **Agenda 8, trust and safety :** I add grounding, abstention when the search score is low, a verifier that checks every number came from a tool result, guardrails to stay on topic, protection against prompt injection, and no private data in logs.
- **Agenda 9, testing :** I write unit tests for each tool, 8 to 10 evals including tricky ones, trajectory evals that check which tool was called, and tracing so each run is logged step by step.
- **Agenda 10, basic reliability :** I add simple retries and timeouts, token and cost tracking, and I keep the prompts in git as versioned files.
- **Agenda 11, delivery:** I write the README with the scaling plan, record the video, and prepare my interview notes.

