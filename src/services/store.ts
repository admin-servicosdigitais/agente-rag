import * as fs from "node:fs";
import * as path from "node:path";

export type VectorItem = {
  id: string;
  sourcePath: string;   // caminho do arquivo .md
  chunkIndex: number;   // índice do pedaço
  text: string;         // texto do chunk
  embedding: number[];  // vetor
};

type FileMeta = { mtimeMs: number; chunkCount: number };

type StoreShape = {
  items: VectorItem[];
  files: Record<string, FileMeta>; // por caminho absoluto
};

export class LocalVectorStore {
  private dir = path.resolve(".rag");
  private file = path.join(this.dir, "vectorstore.json");
  private data: StoreShape = { items: [], files: {} };

  constructor() {
    this.load();
  }

  private ensureDir() {
    if (!fs.existsSync(this.dir)) fs.mkdirSync(this.dir, { recursive: true });
  }

  private load() {
    this.ensureDir();
    if (fs.existsSync(this.file)) {
      try {
        this.data = JSON.parse(fs.readFileSync(this.file, "utf-8"));
      } catch {
        this.data = { items: [], files: {} };
      }
    }
  }

  private save() {
    this.ensureDir();
    fs.writeFileSync(this.file, JSON.stringify(this.data, null, 2), "utf-8");
  }

  needsUpdate(absPath: string, mtimeMs: number): boolean {
    const meta = this.data.files[absPath];
    return !meta || meta.mtimeMs !== mtimeMs;
  }

  upsertFileEmbeddings(
    absPath: string,
    mtimeMs: number,
    chunks: { text: string; embedding: number[]; chunkIndex: number }[]
  ) {
    // remove antigos
    this.data.items = this.data.items.filter((it) => it.sourcePath !== absPath);

    // insere novos
    for (const ch of chunks) {
      this.data.items.push({
        id: `${absPath}::${ch.chunkIndex}`,
        sourcePath: absPath,
        chunkIndex: ch.chunkIndex,
        text: ch.text,
        embedding: ch.embedding,
      });
    }

    this.data.files[absPath] = { mtimeMs, chunkCount: chunks.length };
    this.save();
  }

  stats() {
    const files = Object.keys(this.data.files).length;
    const chunks = this.data.items.length;
    return { files, chunks };
  }

  queryTopK(queryVec: number[], k = 6) {
    const scored = this.data.items.map((it) => ({
      item: it,
      score: cosine(queryVec, it.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  allItems(): VectorItem[] {
    return this.data.items.slice();
  }
}

// ===== Similaridade (cosine) =====
function cosine(a: number[], b: number[]) {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
