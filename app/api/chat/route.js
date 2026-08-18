import Anthropic from "@anthropic-ai/sdk";

const SYSTEM_PROMPT = `You are "Yatra AI" — a premium AI travel assistant for Travel Threads, specializing in India travel planning.

Your personality:
- Warm, knowledgeable, and enthusiastic about Indian travel
- Give specific, actionable advice (not vague generalities)
- Use occasional emojis to make responses feel lively but not overwhelming
- Keep responses concise yet rich in detail — 3–5 short paragraphs max
- Always end with a helpful follow-up question or suggestion

Your expertise:
- Destinations: Manali, Shimla, Jaipur, Rishikesh, Varanasi, Goa, Meghalaya, Ladakh, Kedarnath, Madhya Maheshwar, Dharamshala, Ujjain (and all of India)
- Itinerary planning: day-by-day activities, best routes, travel durations
- Budget planning: budget/mid-range/luxury tiers, cost breakdowns, money-saving tips
- Best times to visit each destination (weather, festivals, crowd levels)
- Local food recommendations, hidden gems, cultural etiquette
- Group travel and expense splitting advice
- Transportation: trains, flights, road trips, local transport

Rules:
- Only assist with travel-related queries. If asked something unrelated, politely redirect.
- Never make up specific prices — give approximate ranges and recommend verifying current rates.
- Be encouraging and inspire wanderlust!`;

// ----------------------------------------------------------------------------
// Config resolution
// ----------------------------------------------------------------------------
// Two task tiers:
//   "fast"  -> quick chat replies, low latency, cheap  (default)
//   "smart" -> deeper planning, more reasoning, higher cost
//
// Each provider maps a tier to a model via env. See .env.local.example.

const PROVIDER = (process.env.AI_PROVIDER || "claude").toLowerCase();

const MODEL_MAP = {
  claude: {
    fast: process.env.MODEL_CLAUDE_FAST || "claude-haiku-4-5",
    smart: process.env.MODEL_CLAUDE_SMART || "claude-haiku-4-5",
  },
  groq: {
    fast: process.env.MODEL_GROQ_FAST || "llama-3.1-8b-instant",
    smart: process.env.MODEL_GROQ_SMART || "llama-3.3-70b-versatile",
  },
};

// Optional Claude reasoning effort per tier — only used on models that support
// it (Opus / Sonnet 4.6+). Blank = don't send the parameter (correct for Haiku).
const CLAUDE_EFFORT = {
  fast: process.env.CLAUDE_EFFORT_FAST || "",
  smart: process.env.CLAUDE_EFFORT_SMART || "",
};

function resolveTier(requested) {
  const tier = (requested || "fast").toLowerCase();
  return tier === "smart" ? "smart" : "fast";
}

// ----------------------------------------------------------------------------
// Provider adapters
// ----------------------------------------------------------------------------

async function callClaude({ system, messages, tier }) {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error(
      "ANTHROPIC_API_KEY missing. Add it to .env.local, or switch AI_PROVIDER to 'groq'."
    );
  }
  const client = new Anthropic();
  const model = MODEL_MAP.claude[tier];
  const effort = CLAUDE_EFFORT[tier];

  const params = {
    model,
    max_tokens: tier === "smart" ? 2048 : 1024,
    system,
    messages,
  };
  // Only send output_config.effort when explicitly configured and the model
  // supports it — Haiku 4.5 will 400 if you send effort.
  if (effort && !model.includes("haiku")) {
    params.output_config = { effort };
  }

  const response = await client.messages.create(params);
  return response.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("");
}

async function callGroq({ system, messages, tier }) {
  if (!process.env.GROQ_API_KEY) {
    throw new Error(
      "GROQ_API_KEY missing. Add it to .env.local, or switch AI_PROVIDER to 'claude'."
    );
  }
  const model = MODEL_MAP.groq[tier];
  const formatted = [{ role: "system", content: system }, ...messages];
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model,
      messages: formatted,
      temperature: 0.7,
      max_tokens: tier === "smart" ? 2048 : 1024,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error?.message || `Groq API returned status ${res.status}`);
  }
  return data.choices?.[0]?.message?.content || "";
}

// ----------------------------------------------------------------------------
// Route handler
// ----------------------------------------------------------------------------

export async function POST(req) {
  let tier = "fast";
  try {
    const body = await req.json();
    const { message, history, taskType } = body;
    tier = resolveTier(taskType);

    if (!message?.trim()) {
      return Response.json({ error: "Message is required" }, { status: 400 });
    }

    const messages = [
      ...(history || []).map((msg) => ({
        role: msg.role === "user" ? "user" : "assistant",
        content: msg.content,
      })),
      { role: "user", content: message },
    ];

    let reply;
    switch (PROVIDER) {
      case "groq":
        reply = await callGroq({ system: SYSTEM_PROMPT, messages, tier });
        break;
      case "claude":
        reply = await callClaude({ system: SYSTEM_PROMPT, messages, tier });
        break;
      default:
        return Response.json(
          {
            error: `Unknown AI_PROVIDER "${PROVIDER}". Use "claude" or "groq" in .env.local.`,
          },
          { status: 500 }
        );
    }

    if (!reply) {
      throw new Error(`No response received from ${PROVIDER} API`);
    }

    return Response.json({
      reply,
      provider: PROVIDER,
      tier,
      model: MODEL_MAP[PROVIDER][tier],
    });
  } catch (err) {
    console.error(`AI provider (${PROVIDER}, tier=${tier}) error:`, err);
    return Response.json(
      { error: err.message || "Failed to get response. Please try again." },
      { status: 500 }
    );
  }
}
