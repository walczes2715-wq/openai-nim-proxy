
"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;
const NIM_API_BASE = (
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NIM_API_KEY =
  process.env.NVIDIA_API_KEY ||
  process.env.NIM_API_KEY;

const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY || "";
const MAX_TOKENS_LIMIT = Number(
  process.env.MAX_TOKENS_LIMIT || 65536
);

const REQUEST_TIMEOUT = 180000;
const REASONING_TIMEOUT = 480000;

const aliases = {
  "gpt-4-turbo": "moonshotai/kimi-k3",
  "gpt-4": "nvidia/nemotron-3-ultra-550b-a55b",
  "gpt-4-flash": "deepseek-ai/deepseek-v4-flash",
  "gpt-4o": "deepseek-ai/deepseek-v4-pro",
  "gpt-3.5o": "nvidia/nemotron-3.5-lightning",
  "gemini-pro": "nvidia/nemotron-70b",
  "claude-3-opus": "nvidia/llama-3.3-nemotron-super-49b-v1",
  "claude-3-sonnet": "openai/gpt-oss-20b",
  "google-light": "google/gemma-4-31b-it",
  "google-lighter": "nvidia/llama-3.1-nemotron-nano-8b-v1",
  "glm-5.3": "z-ai/glm5"
};

const defaultModel =
  process.env.DEFAULT_MODEL || "google/gemma-4-31b-it";

app.disable("x-powered-by");
app.use(cors());
app.use(express.json({ limit: "200mb" }));
app.use(express.urlencoded({
  extended: true,
  limit: "200mb"
}));

// Log only serializable error fields. Never JSON.stringify(error).
function safeError(error) {
  return {
    message: String(error?.message || "Unknown error"),
    status: error?.response?.status || null,
    data: error?.response?.data ?? null,
    code: error?.code || null
  };
}

function sendError(res, status, message, type = "server_error") {
  if (res.headersSent) return;
  res.status(status).json({
    error: {
      message,
      type,
      code: status
    }
  });
}

function authenticate(req, res, next) {
  if (!CLIENT_AUTH_KEY) return next();

  const auth = req.headers.authorization || "";
  const supplied = auth.startsWith("Bearer ")
    ? auth.slice(7)
    : "";

  if (supplied !== CLIENT_AUTH_KEY) {
    return sendError(res, 401, "Invalid client API key", "auth_error");
  }

  next();
}

function resolveModel(model) {
  return aliases[model] || model || defaultModel;
}

function normalizeBody(body) {
  const normalized = { ...body };
  normalized.model = resolveModel(body.model);

  if (typeof normalized.max_tokens === "number") {
    normalized.max_tokens = Math.min(
      normalized.max_tokens,
      MAX_TOKENS_LIMIT
    );
  }

  if (typeof normalized.max_completion_tokens === "number") {
    normalized.max_completion_tokens = Math.min(
      normalized.max_completion_tokens,
      MAX_TOKENS_LIMIT
    );
  }

  // Do not forward proxy-specific fields to NVIDIA.
  delete normalized.client;
  delete normalized.debug;

  return normalized;
}

app.get("/", (req, res) => {
  res.json({
    name: "OpenAI to NVIDIA NIM Proxy",
    version: "2.6.1",
    status: "ok"
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    version: "2.6.1",
    provider: "nvidia-nim-proxy"
  });
});

app.get("/v1/models", authenticate, (req, res) => {
  const models = [
    ...new Set([
      ...Object.keys(aliases),
      ...Object.values(aliases),
      defaultModel
    ])
  ];

  res.json({
    object: "list",
    data: models.map(id => ({
      id,
      object: "model",
      created: 0,
      owned_by: "nim-proxy"
    }))
  });
});

app.post(
  "/v1/chat/completions",
  authenticate,
  async (req, res) => {
    if (!NIM_API_KEY) {
      return sendError(
        res,
        500,
        "Missing NVIDIA_API_KEY or NIM_API_KEY in Railway variables"
      );
    }

    const originalModel = req.body?.model || defaultModel;
    const targetModel = resolveModel(originalModel);

    const body = normalizeBody(req.body || {});
    body.model = targetModel;

    const isKimi = targetModel === "moonshotai/kimi-k3";
    const isReasoning =
      Boolean(body.reasoning_effort) ||
      Boolean(process.env.ENABLE_THINKING_MODE === "true");

    const timeout = isReasoning
      ? REASONING_TIMEOUT
      : REQUEST_TIMEOUT;

    console.log("[NIM] Incoming chat request", {
      requestedModel: originalModel,
      targetModel,
      stream: Boolean(body.stream),
      kimi: isKimi,
      timeout
    });

    try {
      const upstream = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        body,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            "Content-Type": "application/json",
            Accept: body.stream
              ? "text/event-stream"
              : "application/json"
          },
          timeout,
          responseType: body.stream ? "stream" : "json",
          validateStatus: () => true,
          maxBodyLength: Infinity,
          maxContentLength: Infinity
        }
      );

      if (upstream.status < 200 || upstream.status >= 300) {
        let upstreamData = upstream.data;

        if (body.stream && upstreamData) {
          upstreamData = await new Promise(resolve => {
            let collected = "";
            upstreamData.setEncoding("utf8");
            upstreamData.on("data", chunk => {
              collected += chunk;
              if (collected.length > 100000) {
                upstreamData.destroy();
                resolve(collected.slice(0, 100000));
              }
            });
            upstreamData.on("end", () => resolve(collected));
            upstreamData.on("error", () => resolve(collected));
          });
        }

        console.error("[NIM] Upstream returned error", {
          requestedModel: originalModel,
          targetModel,
          status: upstream.status,
          data: upstreamData
        });

        if (body.stream) {
          return sendError(
            res,
            upstream.status,
            typeof upstreamData === "string"
              ? upstreamData
              : JSON.stringify(upstreamData),
            "upstream_error"
          );
        }

        return res.status(upstream.status).json(
          upstreamData || {
            error: { message: "NVIDIA NIM request failed" }
          }
        );
      }

      res.status(upstream.status);

      if (body.stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        upstream.data.on("error", error => {
          console.error("[NIM] Kimi/stream response error", {
            model: targetModel,
            ...safeError(error)
          });

          if (!res.headersSent) {
            sendError(res, 502, "Upstream stream failed");
          } else {
            res.end();
          }
        });

        req.on("close", () => {
          if (!res.writableEnded) upstream.data.destroy();
        });

        return upstream.data.pipe(res);
      }

      return res.json(upstream.data);
    } catch (error) {
      const details = safeError(error);

      console.error("[NIM] Chat request failed", {
        requestedModel: originalModel,
        targetModel,
        ...details
      });

      if (res.headersSent) {
        return res.end();
      }

      const status =
        error?.response?.status ||
        (error.code === "ECONNABORTED" ? 504 : 502);

      return sendError(
        res,
        status,
        details.data?.error?.message ||
          details.data?.message ||
          details.message ||
          "NVIDIA NIM request failed",
        "upstream_error"
      );
    }
  }
);

// Body-parser errors and other Express errors.
app.use((error, req, res, next) => {
  console.error("[Express] Request error", safeError(error));

  if (res.headersSent) return next(error);

  const status = error.status || 500;
  return sendError(
    res,
    status,
    status === 413
      ? "Request body too large"
      : error.message || "Internal server error"
  );
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[NIM Proxy] Listening on port ${PORT}`);
  console.log(`[NIM Proxy] NVIDIA endpoint: ${NIM_API_BASE}`);
  console.log(`[NIM Proxy] Client authentication: ${
    CLIENT_AUTH_KEY ? "enabled" : "disabled"
  }`);
});
