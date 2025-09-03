import { LocalVectorStore } from "./store";
import { chatWithContext } from "./bedrock";

export interface RetrievalOptions {
  topK?: number;
  maxContextChars?: number;
}

const DEFAULT_SYSTEM_PROMPT = [
  "Você é um agente RAG.",
  "Use APENAS os trechos do CONTEXTO para responder.",
  "Se a resposta não estiver presente no contexto, diga explicitamente: 'Não encontrei essa informação nos documentos fornecidos.'",
  "Seja conciso e direto.",
].join("\n");

export async function answerWithRAG(
  store: LocalVectorStore,
  queryVec: number[],
  question: string,
  opts: RetrievalOptions = {}
): Promise<{ answer: string; used: { sourcePath: string; chunkIndex: number }[] }> {
  const topK = opts.topK ?? Number(process.env.TOP_K ?? 6);
  const maxCtx = opts.maxContextChars ?? Number(process.env.MAX_CONTEXT_CHARS ?? 6000);

  // busca topK
  const results = store.queryTopK(queryVec, topK);

  // monta contexto concatenado (limitando o total de chars)
  let ctx = "";
  const used: { sourcePath: string; chunkIndex: number }[] = [];
  for (const r of results) {
    const header = `Fonte: ${r.item.sourcePath} (chunk ${r.item.chunkIndex}) [score=${r.score.toFixed(3)}]`;
    const block = `### ${header}\n${r.item.text}\n\n---\n`;
    if (ctx.length + block.length > maxCtx) break;
    ctx += block;
    used.push({ sourcePath: r.item.sourcePath, chunkIndex: r.item.chunkIndex });
  }

  const userText = [
    "Responda à PERGUNTA utilizando exclusivamente o CONTEXTO a seguir.",
    "Se não houver informação suficiente, diga que não encontrou nos documentos.",
    "",
    "=== CONTEXTO ===",
    ctx || "(vazio)",
    "=== FIM DO CONTEXTO ===",
    "",
    `PERGUNTA: ${question}`,
  ].join("\n");

  const answer = await chatWithContext(DEFAULT_SYSTEM_PROMPT, userText);
  return { answer, used };
}
