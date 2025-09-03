import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";

// ===== Config =====
export const REGION = process.env.AWS_REGION ?? "us-east-1";
export const CHAT_MODEL_ID = process.env.BEDROCK_MODEL_ID ?? "anthropic.claude-3-haiku-20240307-v1:0";
export const EMBED_MODEL_ID = process.env.BEDROCK_EMBED_MODEL_ID ?? "amazon.titan-embed-text-v2:0";
export const MAX_TOKENS = Number(process.env.MAX_TOKENS ?? 512);
export const TEMPERATURE = Number(process.env.TEMPERATURE ?? 0.1);

const client = new BedrockRuntimeClient({ region: REGION });

// ===== Embeddings (Titan) =====
export async function embedText(text: string): Promise<number[]> {
  const body = JSON.stringify({ inputText: text });
  const res = await client.send(
    new InvokeModelCommand({
      modelId: EMBED_MODEL_ID,
      contentType: "application/json",
      accept: "application/json",
      body,
    })
  );

  const json = JSON.parse(new TextDecoder().decode(res.body));
  // v2 costuma retornar "embedding"; também lidamos com "embeddings[0].embedding"
  const vec: number[] =
    json?.embedding ??
    (Array.isArray(json?.embeddings) ? json.embeddings[0]?.embedding : undefined);

  if (!Array.isArray(vec)) {
    throw new Error("Falha ao obter embedding do Titan (resposta inesperada).");
  }
  return vec;
}

// ===== Chat (Anthropic via Bedrock) =====
export async function chatWithContext(systemPrompt: string | undefined, userText: string): Promise<string> {
  const body = {
    anthropic_version: "bedrock-2023-05-31",
    max_tokens: MAX_TOKENS,
    temperature: TEMPERATURE,
    system: systemPrompt,
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: userText }],
      },
    ],
  };

  const res = await client.send(
    new InvokeModelCommand({
      modelId: CHAT_MODEL_ID,
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify(body),
    })
  );

  const json = JSON.parse(new TextDecoder().decode(res.body));
  const text =
    Array.isArray(json.content) && json.content[0]?.text
      ? json.content[0].text
      : JSON.stringify(json);
  return text as string;
}
