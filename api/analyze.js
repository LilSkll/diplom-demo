const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type"
};

const OPENAI_URL = "https://api.openai.com/v1/chat/completions";
const QWEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation";
const DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions";
const SUPPORTED_LANGS = ["en", "es", "de", "ru"];
const SUPPORTED_MODES = ["morph", "translate", "compare"];

function sendJson(res, status, payload) {
  res.status(status);
  Object.entries(CORS_HEADERS).forEach(([key, value]) => res.setHeader(key, value));
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.send(JSON.stringify(payload));
}

function normalizeJsonString(content) {
  if (typeof content !== "string") return "";
  return content.trim().replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/```$/i, "").trim();
}

function buildPrompt({ text, lang, mode }) {
  if (mode === "morph") {
    return `You are a multilingual computational linguistics engine.
Analyze the input text morphologically and syntactically.
Language code: ${lang}.

Return ONLY valid JSON. No markdown, no explanation, no extra text.
Use this exact schema and keys:
{
  "tokens": [
    {
      "word": "original word",
      "lemma": "base form",
      "pos": "NOUN|VERB|ADJ|ADV|DET|ADP|PRON|AUX|CCONJ|PUNCT|NUM",
      "dep": "nsubj|ROOT|obj|det|amod|prep|nmod|punct|...",
      "head": "word this token depends on",
      "morph": "Case=Nom|Number=Sing|... (Universal Dependencies format)"
    }
  ],
  "synth_index": "1.68",
  "model_name": "gpt-4o-mini (${lang})",
  "sentence_count": 1
}

Rules:
- Keep token order exactly as in the input.
- Use ROOT for the syntactic root token dep.
- If token has no morphological features, set "morph" to "".
- sentence_count must be the number of sentences in input text.

Text:
${text}`;
  }

  if (mode === "translate") {
    return `You are a professional multilingual translator.
Source language code: ${lang}.

Return ONLY valid JSON. No markdown, no explanation, no extra text.
{
  "translations": {
    "en": "English text",
    "es": "Spanish translation",
    "de": "German translation",
    "ru": "Russian translation"
  },
  "source_lang": "${lang}"
}

Text:
${text}`;
  }

  return `You are a multilingual linguistics analysis engine.
Given the input text and source language code "${lang}", create semantically equivalent versions in English, Spanish, German, and Russian.
For each language, provide token-level morphology and dependencies.

Return ONLY valid JSON. No markdown, no explanation, no extra text.
Use this schema:
{
  "compare": {
    "en": { "text": "...", "tokens": [ { "word": "...", "lemma": "...", "pos": "...", "dep": "...", "head": "...", "morph": "..." } ] },
    "es": { "text": "...", "tokens": [ { "word": "...", "lemma": "...", "pos": "...", "dep": "...", "head": "...", "morph": "..." } ] },
    "de": { "text": "...", "tokens": [ { "word": "...", "lemma": "...", "pos": "...", "dep": "...", "head": "...", "morph": "..." } ] },
    "ru": { "text": "...", "tokens": [ { "word": "...", "lemma": "...", "pos": "...", "dep": "...", "head": "...", "morph": "..." } ] }
  }
}

Text:
${text}`;
}

async function callOpenAI({ apiKey, prompt }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 25000);

  const response = await fetch(OPENAI_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.1,
      max_tokens: 1800,
      messages: [
        {
          role: "system",
          content: "You always return strict JSON only."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      response_format: { type: "json_object" }
    }),
    signal: controller.signal
  }).finally(() => {
    clearTimeout(timeout);
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenAI request failed: ${response.status} ${errorBody}`);
  }

  const body = await response.json();
  const content = body?.choices?.[0]?.message?.content;
  const normalized = normalizeJsonString(content);

  if (!normalized) {
    throw new Error("OpenAI returned empty response content.");
  }

  return JSON.parse(normalized);
}

async function callDeepSeek({ apiKey, prompt }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 25000);

  const response = await fetch(DEEPSEEK_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: "deepseek-chat",
      temperature: 0.1,
      max_tokens: 1800,
      messages: [
        {
          role: "system",
          content: "You always return strict JSON only."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      response_format: { type: "json_object" }
    }),
    signal: controller.signal
  }).finally(() => {
    clearTimeout(timeout);
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`DeepSeek request failed: ${response.status} ${errorBody}`);
  }

  const body = await response.json();
  const content = body?.choices?.[0]?.message?.content;
  const normalized = normalizeJsonString(content);

  if (!normalized) {
    throw new Error("DeepSeek returned empty response content.");
  }

  return JSON.parse(normalized);
}

async function callAIWithFallback({ prompt }) {
  const providers = [
    {
      name: "OpenAI",
      key: process.env.OPENAI_API_KEY,
      func: callOpenAI
    },
    {
      name: "DeepSeek",
      key: process.env.DEEPSEEK_API_KEY,
      func: callDeepSeek
    }
  ];

  for (const provider of providers) {
    if (!provider.key) {
      console.log(`${provider.name} API key not configured, skipping...`);
      continue;
    }

    try {
      console.log(`Trying ${provider.name}...`);
      return await provider.func({ apiKey: provider.key, prompt });
    } catch (error) {
      console.error(`${provider.name} failed:`, error.message);
      continue;
    }
  }

  throw new Error("All AI providers failed. Please configure at least one API key.");
}

function validatePayload(payload) {
  if (!payload || typeof payload !== "object") return "Body must be a JSON object.";
  if (typeof payload.text !== "string" || !payload.text.trim()) return "Field 'text' must be a non-empty string.";
  if (typeof payload.lang !== "string" || !SUPPORTED_LANGS.includes(payload.lang)) return "Field 'lang' must be one of: en, es, de, ru.";
  if (typeof payload.mode !== "string" || !SUPPORTED_MODES.includes(payload.mode)) return "Field 'mode' must be one of: morph, translate, compare.";
  return null;
}

export default async function handler(req, res) {
  Object.entries(CORS_HEADERS).forEach(([key, value]) => res.setHeader(key, value));

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  if (req.method !== "POST") {
    sendJson(res, 405, { error: "Method not allowed. Use POST." });
    return;
  }

  const hasOpenAI = !!process.env.OPENAI_API_KEY;
  const hasDeepSeek = !!process.env.DEEPSEEK_API_KEY;
  
  console.log("Available providers:", {
    OpenAI: hasOpenAI,
    DeepSeek: hasDeepSeek
  });
  
  if (!hasOpenAI && !hasDeepSeek) {
    sendJson(res, 500, { 
      error: "No AI provider configured. Please add OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable." 
    });
    return;
  }

  let payload;
  try {
    payload = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
  } catch {
    sendJson(res, 400, { error: "Invalid JSON body." });
    return;
  }

  const validationError = validatePayload(payload);
  if (validationError) {
    sendJson(res, 400, { error: validationError });
    return;
  }

  try {
    const prompt = buildPrompt(payload);
    const aiResult = await callAIWithFallback({ prompt });

    if (payload.mode === "morph") {
      sendJson(res, 200, {
        tokens: Array.isArray(aiResult.tokens) ? aiResult.tokens : [],
        synth: String(aiResult.synth_index ?? "0.00"),
        model: String(aiResult.model_name ?? `gpt-4o-mini (${payload.lang})`),
        sentence_count: Number(aiResult.sentence_count ?? 1)
      });
      return;
    }

    if (payload.mode === "translate") {
      const translations = aiResult.translations || {};
      sendJson(res, 200, {
        translations: {
          en: String(translations.en ?? ""),
          es: String(translations.es ?? ""),
          de: String(translations.de ?? ""),
          ru: String(translations.ru ?? "")
        },
        source_lang: String(aiResult.source_lang ?? payload.lang)
      });
      return;
    }

    sendJson(res, 200, {
      compare: aiResult.compare || {}
    });
    return;
  } catch (error) {
    console.error("API Error:", error);
    console.error("Error message:", error instanceof Error ? error.message : "Unknown error");
    console.error("Stack:", error instanceof Error ? error.stack : "No stack");
    
    sendJson(res, 500, {
      error: "Failed to analyze text with OpenAI.",
      details: error instanceof Error ? error.message : "Unknown server error."
    });
    return;
  }
}
