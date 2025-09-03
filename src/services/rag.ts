import * as fs from "node:fs/promises";
import * as fssync from "node:fs";
import * as path from "node:path";
import { LocalVectorStore } from "./store";
import { embedText } from "./bedrock";

// Pasta com fontes .md
export const SOURCES_DIR = path.resolve("./fontes");

export async function ensureSourcesDir() {
  if (!fssync.existsSync(SOURCES_DIR)) {
    await fs.mkdir(SOURCES_DIR, { recursive: true });
  }
}

/**
 * Atualiza o índice local:
 * - varre a pasta ./fontes por arquivos .md
 * - para cada arquivo, se modificado: faz chunking + embeddings e grava no store
 */
export async function updateIndex(store: LocalVectorStore) {
  await ensureSourcesDir();
  const dirents = await fs.readdir(SOURCES_DIR, { withFileTypes: true });
  const mdFiles = dirents
    .filter((d) => d.isFile() && d.name.toLowerCase().endsWith(".md"))
    .map((d) => path.join(SOURCES_DIR, d.name));

  let processed = 0;
  let skipped = 0;
  let totalChunks = 0;

  for (const filePath of mdFiles) {
    const abs = path.resolve(filePath);
    const st = await fs.stat(abs);

    if (!store.needsUpdate(abs, st.mtimeMs)) {
      skipped++;
      continue;
    }

    const content = await fs.readFile(abs, "utf-8");
    const chunks = splitMarkdownToChunks(content);

    const embedded: { text: string; embedding: number[]; chunkIndex: number }[] = [];
    for (let i = 0; i < chunks.length; i++) {
      const text = chunks[i];
      const vec = await embedText(text);
      embedded.push({ text, embedding: vec, chunkIndex: i });
    }

    store.upsertFileEmbeddings(abs, st.mtimeMs, embedded);

    processed++;
    totalChunks += chunks.length;
  }

  return { processed, skipped, totalChunks, stats: store.stats() };
}

// === Chunker simples para Markdown ===
// Estratégia: 
// 1) split por seções (#, ##, ###)  
// 2) fatiar blocos longos por tamanho com overlap
export function splitMarkdownToChunks(
  content: string,
  maxChars = Number(process.env.CHUNK_CHARS ?? 1500),
  overlap = Number(process.env.CHUNK_OVERLAP ?? 200)
): string[] {
  const sections = content
    .split(/\n(?=#+\s)/g) // quebra quando encontra um cabeçalho no início da linha
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const raw = sections.length > 0 ? sections : [content];

  const result: string[] = [];
  for (const sec of raw) {
    if (sec.length <= maxChars) {
      result.push(sec);
      continue;
    }
    // fatiamento com overlap
    let start = 0;
    while (start < sec.length) {
      const end = Math.min(sec.length, start + maxChars);
      const slice = sec.slice(start, end);
      result.push(slice);
      if (end >= sec.length) break;
      start = end - overlap; // recuo para overlap
      if (start < 0) start = 0;
    }
  }

  // remove linhas vazias muito longas e normaliza
  return result.map((t) => t.replace(/\n{3,}/g, "\n\n").trim()).filter(Boolean);
}
